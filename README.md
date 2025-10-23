# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-22 17:00:25.761342 PST.

### Artificial Intelligence

### 1. [ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](http://arxiv.org/pdf/2510.18250v1)

Authors: Xiaohan Qin, Xiaoxing Wang, Ning Liao, Cancheng Zhang, Xiangdong Zhang, Mingquan Feng, Jingzhi Wang, Junchi Yan

Data quality plays a critical role in enhancing supervised fine-tuning (SFT)
for large language models (LLMs), and token-level data selection has emerged as
a promising direction for its fine-grained nature. Despite their strong
empirical performance, existing token-level selection methods share two key
limitations: (1) requiring training or accessing an additional reference model,
and (2) relying solely on loss information for token selection, which cannot
well preserve semantically important tokens that are not favored by loss-based
metrics. To address these challenges, we propose ssToken, a Self-modulated and
Semantic-aware Token Selection approach. ssToken leverages readily accessible
history models to compute the per-token loss difference with the current model,
which serves as a self-modulated signal that enables the model to adaptively
select tokens along its optimization trajectory, rather than relying on excess
loss from an offline-trained reference model as in prior works. We further
introduce a semantic-aware, attention-based token importance estimation metric,
orthogonal to loss-based selection and providing complementary semantic
information for more effective filtering. Extensive experiments across
different model families and scales demonstrate that both self-modulated
selection and semantic-aware selection alone outperform full-data fine-tuning,
while their integration--ssToken--achieves synergistic gains and further
surpasses prior token-level selection methods, delivering performance
improvements while maintaining training efficiency.

### 2. [Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming](http://arxiv.org/pdf/2510.18314v1)

Authors: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

As large language model (LLM) agents increasingly automate complex web tasks,
they boost productivity while simultaneously introducing new security risks.
However, relevant studies on web agent attacks remain limited. Existing
red-teaming approaches mainly rely on manually crafted attack strategies or
static models trained offline. Such methods fail to capture the underlying
behavioral patterns of web agents, making it difficult to generalize across
diverse environments. In web agent attacks, success requires the continuous
discovery and evolution of attack strategies. To this end, we propose Genesis,
a novel agentic framework composed of three modules: Attacker, Scorer, and
Strategist. The Attacker generates adversarial injections by integrating the
genetic algorithm with a hybrid strategy representation. The Scorer evaluates
the target web agent's responses to provide feedback. The Strategist
dynamically uncovers effective strategies from interaction logs and compiles
them into a continuously growing strategy library, which is then re-deployed to
enhance the Attacker's effectiveness. Extensive experiments across various web
tasks show that our framework discovers novel strategies and consistently
outperforms existing attack baselines.

### 3. [Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning](http://arxiv.org/pdf/2510.18318v1)

Authors: Aaron Bell, Amit Aides, Amr Helmy, Arbaaz Muslim, Aviad Barzilai, Aviv Slobodkin, Bolous Jaber, David Schottlander, George Leifman, Joydeep Paul, Mimi Sun, Nadav Sherman, Natalie Williams, Per Bjornsson, Roy Lee, Ruth Alcantara, Thomas Turnbull, Tomer Shekel, Vered Silverman, Yotam Gigi, Adam Boulanger, Alex Ottenwess, Ali Ahmadalipour, Anna Carter, Charles Elliott, David Andre, Elad Aharoni, Gia Jung, Hassler Thurston, Jacob Bien, Jamie McPike, Juliet Rothenberg, Kartik Hegde, Kel Markert, Kim Philipp Jablonski, Luc Houriez, Monica Bharel, Phing VanLee, Reuven Sayag, Sebastian Pilarski, Shelley Cazares, Shlomi Pasternak, Siduo Jiang, Stone Jiang, Thomas Colthurst, Yang Chen, Yehonathan Refael, Yochai Blau, Yuval Carny, Yael Maguire, Avinatan Hassidim, James Manyika, Tim Thelin, Genady Beryozkin, Gautam Prasad, Luke Barrington, Yossi Matias, Niv Efron, Shravya Shetty

Geospatial data offers immense potential for understanding our planet.
However, the sheer volume and diversity of this data along with its varied
resolutions, timescales, and sparsity pose significant challenges for thorough
analysis and interpretation. This paper introduces Earth AI, a family of
geospatial AI models and agentic reasoning that enables significant advances in
our ability to unlock novel and profound insights into our planet. This
approach is built upon foundation models across three key domains--Planet-scale
Imagery, Population, and Environment--and an intelligent Gemini-powered
reasoning engine. We present rigorous benchmarks showcasing the power and novel
capabilities of our foundation models and validate that when used together,
they provide complementary value for geospatial inference and their synergies
unlock superior predictive capabilities. To handle complex, multi-step queries,
we developed a Gemini-powered agent that jointly reasons over our multiple
foundation models along with large geospatial data sources and tools. On a new
benchmark of real-world crisis scenarios, our agent demonstrates the ability to
deliver critical and timely insights, effectively bridging the gap between raw
geospatial data and actionable understanding.

### 4. [ShortcutBreaker: Low-Rank Noisy Bottleneck with Global Perturbation Attention for Multi-Class Unsupervised Anomaly Detection](http://arxiv.org/pdf/2510.18342v1)

Authors: Peng Tang, Xiaoxiao Yan, Xiaobin Hu, Yuning Cui, Donghao Luo, Jiangning Zhang, Pengcheng Xu, Jinlong Peng, Qingdong He, Feiyue Huang, Song Xue, Tobias Lasser

Multi-class unsupervised anomaly detection (MUAD) has garnered growing
research interest, as it seeks to develop a unified model for anomaly detection
across multiple classes, i.e., eliminating the need to train separate models
for distinct objects and thereby saving substantial computational resources.
Under the MUAD setting, while advanced Transformer-based architectures have
brought significant performance improvements, identity shortcuts persist: they
directly copy inputs to outputs, narrowing the gap in reconstruction errors
between normal and abnormal cases, and thereby making the two harder to
distinguish. Therefore, we propose ShortcutBreaker, a novel unified
feature-reconstruction framework for MUAD tasks, featuring two key innovations
to address the issue of shortcuts. First, drawing on matrix rank inequality, we
design a low-rank noisy bottleneck (LRNB) to project highdimensional features
into a low-rank latent space, and theoretically demonstrate its capacity to
prevent trivial identity reproduction. Second, leveraging ViTs global modeling
capability instead of merely focusing on local features, we incorporate a
global perturbation attention to prevent information shortcuts in the decoders.
Extensive experiments are performed on four widely used anomaly detection
benchmarks, including three industrial datasets (MVTec-AD, ViSA, and Real-IAD)
and one medical dataset (Universal Medical). The proposed method achieves a
remarkable image-level AUROC of 99.8%, 98.9%, 90.6%, and 87.8% on these four
datasets, respectively, consistently outperforming previous MUAD methods across
different scenarios.

### 5. [Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games](http://arxiv.org/pdf/2510.18395v1)

Authors: Runnan Qi, Yanan Ni, Lumin Jiang, Zongyuan Li, Kuihua Huang, Xian Guo

This paper proposes Memory-Augmented State Machine Prompting (MASMP), a novel
framework for LLM agents in real-time strategy games. Addressing key challenges
like hallucinations and fragmented decision-making in existing approaches,
MASMP integrates state machine prompting with memory mechanisms to unify
structured actions with long-term tactical coherence. The framework features:
(1) a natural language-driven state machine architecture that guides LLMs to
emulate finite state machines and behavior trees through prompts, and (2) a
lightweight memory module preserving strategic variables (e.g., tactics,
priority units) across decision cycles. Experiments in StarCraft II demonstrate
MASMP's 60% win rate against the hardest built-in AI (Lv7), vastly
outperforming baselines (0%). Case studies reveal the method retains LLMs'
semantic comprehension while resolving the "Knowing-Doing Gap" through strict
state-action mapping, achieving both interpretability and FSM-like reliability.
This work establishes a new paradigm for combining neural and symbolic AI in
complex decision-making.

### 6. [Heterogeneous Adversarial Play in Interactive Environments](http://arxiv.org/pdf/2510.18407v1)

Authors: Manjie Xu, Xinyi Yang, Jiayu Zhan, Wei Liang, Chi Zhang, Yixin Zhu

Self-play constitutes a fundamental paradigm for autonomous skill
acquisition, whereby agents iteratively enhance their capabilities through
self-directed environmental exploration. Conventional self-play frameworks
exploit agent symmetry within zero-sum competitive settings, yet this approach
proves inadequate for open-ended learning scenarios characterized by inherent
asymmetry. Human pedagogical systems exemplify asymmetric instructional
frameworks wherein educators systematically construct challenges calibrated to
individual learners' developmental trajectories. The principal challenge
resides in operationalizing these asymmetric, adaptive pedagogical mechanisms
within artificial systems capable of autonomously synthesizing appropriate
curricula without predetermined task hierarchies. Here we present Heterogeneous
Adversarial Play (HAP), an adversarial Automatic Curriculum Learning framework
that formalizes teacher-student interactions as a minimax optimization wherein
task-generating instructor and problem-solving learner co-evolve through
adversarial dynamics. In contrast to prevailing ACL methodologies that employ
static curricula or unidirectional task selection mechanisms, HAP establishes a
bidirectional feedback system wherein instructors continuously recalibrate task
complexity in response to real-time learner performance metrics. Experimental
validation across multi-task learning domains demonstrates that our framework
achieves performance parity with SOTA baselines while generating curricula that
enhance learning efficacy in both artificial agents and human subjects.

### 7. [Deep Learning-Based Control Optimization for Glass Bottle Forming](http://arxiv.org/pdf/2510.18412v1)

Authors: Mattia Pujatti, Andrea Di Luca, Nicola Peghini, Federico Monegaglia, Marco Cristoforetti

In glass bottle manufacturing, precise control of forming machines is
critical for ensuring quality and minimizing defects. This study presents a
deep learning-based control algorithm designed to optimize the forming process
in real production environments. Using real operational data from active
manufacturing plants, our neural network predicts the effects of parameter
changes based on the current production setup. Through a specifically designed
inversion mechanism, the algorithm identifies the optimal machine settings
required to achieve the desired glass gob characteristics. Experimental results
on historical datasets from multiple production lines show that the proposed
method yields promising outcomes, suggesting potential for enhanced process
stability, reduced waste, and improved product consistency. These results
highlight the potential of deep learning to process control in glass
manufacturing.

### 8. [Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents](http://arxiv.org/pdf/2510.18424v1)

Authors: Guangfu Guo, Xiaoqian Lu, Yue Feng

Visual Language Models (VLMs) achieve promising results in medical reasoning
but struggle with hallucinations, vague descriptions, inconsistent logic and
poor localization. To address this, we propose a agent framework named Medical
Visual Reasoning Agent (\textbf{Med-VRAgent}). The approach is based on Visual
Guidance and Self-Reward paradigms and Monte Carlo Tree Search (MCTS). By
combining the Visual Guidance with tree search, Med-VRAgent improves the
medical visual reasoning capabilities of VLMs. We use the trajectories
collected by Med-VRAgent as feedback to further improve the performance by
fine-tuning the VLMs with the proximal policy optimization (PPO) objective.
Experiments on multiple medical VQA benchmarks demonstrate that our method
outperforms existing approaches.

### 9. [Automated urban waterlogging assessment and early warning through a mixture of foundation models](http://arxiv.org/pdf/2510.18425v1)

Authors: Chenxu Zhang, Fuxiang Huang, Lei Zhang

With climate change intensifying, urban waterlogging poses an increasingly
severe threat to global public safety and infrastructure. However, existing
monitoring approaches rely heavily on manual reporting and fail to provide
timely and comprehensive assessments. In this study, we present Urban
Waterlogging Assessment (UWAssess), a foundation model-driven framework that
automatically identifies waterlogged areas in surveillance images and generates
structured assessment reports. To address the scarcity of labeled data, we
design a semi-supervised fine-tuning strategy and a chain-of-thought (CoT)
prompting strategy to unleash the potential of the foundation model for
data-scarce downstream tasks. Evaluations on challenging visual benchmarks
demonstrate substantial improvements in perception performance. GPT-based
evaluations confirm the ability of UWAssess to generate reliable textual
reports that accurately describe waterlogging extent, depth, risk and impact.
This dual capability enables a shift of waterlogging monitoring from perception
to generation, while the collaborative framework of multiple foundation models
lays the groundwork for intelligent and scalable systems, supporting urban
management, disaster response and climate resilience.

### 10. [AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library](http://arxiv.org/pdf/2510.18428v1)

Authors: Minwei Kong, Ao Qu, Xiaotong Guo, Wenbin Ouyang, Chonghe Jiang, Han Zheng, Yining Ma, Dingyi Zhuang, Yuhan Tang, Junyi Li, Hai Wang, Cathy Wu, Jinhua Zhao

Optimization modeling enables critical decisions across industries but
remains difficult to automate: informal language must be mapped to precise
mathematical formulations and executable solver code. Prior LLM approaches
either rely on brittle prompting or costly retraining with limited
generalization. We present AlphaOPT, a self-improving experience library that
enables an LLM to learn from limited demonstrations (even answers alone,
without gold-standard programs) and solver feedback - without annotated
reasoning traces or parameter updates. AlphaOPT operates in a continual
two-phase cycle: (i) a Library Learning phase that reflects on failed attempts,
extracting solver-verified, structured insights as {taxonomy, condition,
explanation, example}; and (ii) a Library Evolution phase that diagnoses
retrieval misalignments and refines the applicability conditions of stored
insights, improving transfer across tasks. This design (1) learns efficiently
from limited demonstrations without curated rationales, (2) expands continually
without costly retraining by updating the library rather than model weights,
and (3) makes knowledge explicit and interpretable for human inspection and
intervention. Experiments show that AlphaOPT steadily improves with more data
(65% to 72% from 100 to 300 training items) and surpasses the strongest
baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only
on answers. Code and data are available at:
https://github.com/Minw913/AlphaOPT.

### Hardware Architecture

### 1. [From Quarter to All: Accelerating Speculative LLM Decoding via Floating-Point Exponent Remapping and Parameter Sharing](http://arxiv.org/pdf/2510.18525v1)

Authors: Yushu Zhao, Yubin Qin, Yang Wang, Xiaolong Yang, Huiming Han, Shaojun Wei, Yang Hu, Shouyi Yin

Large language models achieve impressive performance across diverse tasks but
exhibit high inference latency due to their large parameter sizes. While
quantization reduces model size, it often leads to performance degradation
compared to the full model. Speculative decoding remains lossless but typically
incurs extra overheads. We propose SPEQ, an algorithm-hardware co-designed
speculative decoding method that uses part of the full-model weight bits to
form a quantized draft model, thereby eliminating additional training or
storage overhead. A reconfigurable processing element array enables efficient
execution of both the draft and verification passes. Experimental results
across 15 LLMs and tasks demonstrate that SPEQ achieves speedups of 2.07x,
1.53x, and 1.45x compared over FP16, Olive, and Tender, respectively.

### 2. [DRsam: Detection of Fault-Based Microarchitectural Side-Channel Attacks in RISC-V Using Statistical Preprocessing and Association Rule Mining](http://arxiv.org/pdf/2510.18612v1)

Authors: Muhammad Hassan, Maria Mushtaq, Jaan Raik, Tara Ghasempouri

RISC-V processors are becoming ubiquitous in critical applications, but their
susceptibility to microarchitectural side-channel attacks is a serious concern.
Detection of microarchitectural attacks in RISC-V is an emerging research topic
that is relatively underexplored, compared to x86 and ARM. The first line of
work to detect flush+fault-based microarchitectural attacks in RISC-V leverages
Machine Learning (ML) models, yet it leaves several practical aspects that need
further investigation. To address overlooked issues, we leveraged gem5 and
propose a new detection method combining statistical preprocessing and
association rule mining having reconfiguration capabilities to generalize the
detection method for any microarchitectural attack. The performance comparison
with state-of-the-art reveals that the proposed detection method achieves up to
5.15% increase in accuracy, 7% rise in precision, and 3.91% improvement in
recall under the cryptographic, computational, and memory-intensive workloads
alongside its flexibility to detect new variant of flush+fault attack.
Moreover, as the attack detection relies on association rules, their
human-interpretable nature provides deep insight to understand
microarchitectural behavior during the execution of attack and benign
applications.

### 3. [sNVMe-oF: Secure and Efficient Disaggregated Storage](http://arxiv.org/pdf/2510.18756v1)

Authors: Marcin Chrapek, Meni Orenbach, Ahmad Atamli, Marcin Copik, Fritz Alder, Torsten Hoefler

Disaggregated storage with NVMe-over-Fabrics (NVMe-oF) has emerged as the
standard solution in modern data centers, achieving superior performance,
resource utilization, and power efficiency. Simultaneously, confidential
computing (CC) is becoming the de facto security paradigm, enforcing stronger
isolation and protection for sensitive workloads. However, securing
state-of-the-art storage with traditional CC methods struggles to scale and
compromises performance or security. To address these issues, we introduce
sNVMe-oF, a storage management system extending the NVMe-oF protocol and
adhering to the CC threat model by providing confidentiality, integrity, and
freshness guarantees. sNVMe-oF offers an appropriate control path and novel
concepts such as counter-leasing. sNVMe-oF also optimizes data path performance
by leveraging NVMe metadata, introducing a new disaggregated Hazel Merkle Tree
(HMT), and avoiding redundant IPSec protections. We achieve this without
modifying the NVMe-oF protocol. To prevent excessive resource usage while
delivering line rate, sNVMe-oF also uses accelerators of CC-capable smart NICs.
We prototype sNVMe-oF on an NVIDIA BlueField-3 and demonstrate how it can
achieve as little as 2% performance degradation for synthetic patterns and AI
training.

### Computational Complexity

### 1. [Coloring Graphs with Few Colors in the Streaming Model](http://arxiv.org/pdf/2510.18177v1)

Authors: Sepehr Assadi, Janani Sundaresan, Helia Yazdanyar

We study graph coloring problems in the streaming model, where the goal is to
process an $n$-vertex graph whose edges arrive in a stream, using a limited
space that is smaller than the trivial $O(n^2)$ bound. While prior work has
largely focused on coloring graphs with a large number of colors, we explore
the opposite end of the spectrum: deciding whether the input graph can be
colored using only a few, say, a constant number of colors. We are interested
in each of the adversarial, random order, or dynamic streams.
  Our work lays the foundation for this new direction by establishing upper and
lower bounds on space complexity of key variants of the problem. Some of our
main results include:
  - Adversarial: for distinguishing between $q$- vs $2^{\Omega(q)}$-colorable
graphs, lower bounds of $n^{2-o(1)}$ space for $q$ up to
$(\log{n})^{1/2-o(1)}$, and $n^{1+\Omega(1/\log\log{n})}$ space for $q$ further
up to $(\log{n})^{1-o(1)}$.
  - Random order: for distinguishing between $q$- vs $q^t$-colorable graphs for
$q,t \geq 2$, an upper bound of $\tilde{O}(n^{1+1/t})$ space. Specifically,
distinguishing between $q$-colorable graphs vs ones that are not even
poly$(q)$-colorable can be done in $n^{1+o(1)}$ space unlike in adversarial
streams. Although, distinguishing between $q$-colorable vs
$\Omega(q^2)$-colorable graphs requires $\Omega(n^2)$ space even in random
order streams for constant $q$.
  - Dynamic: for distinguishing between $q$- vs $q \cdot t$-colorable graphs
for any $q \geq 3$ and $t \geq 1$, nearly optimal upper and lower bounds of
$\tilde{\Theta}(n^2/t^2)$ space.
  We develop several new technical tools along the way: cluster packing graphs,
a generalization of Ruzsa-Szemer\'edi graphs; a player elimination framework
based on cluster packing graphs; and new edge and vertex sampling lemmas
tailored to graph coloring.

### 2. [Predicative Ordinal Recursion on the Constructive Veblen Hierarchy](http://arxiv.org/pdf/2510.18497v1)

Authors: Amirhossein Akbar Tabatabai, Vitor Greati, Revantha Ramanayake

Inspired by Leivant's work on absolute predicativism, Bellantoni and Cook in
1992 introduced a structurally restricted form of recursion called predicative
recursion. Using this recursion scheme on the inductive structures of natural
numbers and binary strings, they provide a structural and machine-independent
characterization of the classes of linear-space and polynomial-time computable
functions, respectively. This recursion scheme can be applied to any
well-founded or inductive structure, and its underlying principle,
predicativization, extends naturally to other computational frameworks, such as
higher-order functionals and nested recursion.
  In this paper, we initiate a systematic project to gauge the computational
power of predicative recursion on arbitrary well-founded structures. As a
natural measuring stick for well-foundedness, we use constructive ordinals.
More precisely, for any downset $\mathsf{A}$ of constructive ordinals, we
define a class $\mathrm{PredR}_{\mathsf{A}}$ of predicative ordinal recursive
functions that are permitted to employ a suitable form of predicative recursion
on the ordinals in $\mathsf{A}$. We focus on the case that $\mathsf{A}$ is a
downset of constructive ordinals below ${\phi}_{{\omega}}({0}) =
\bigcup_{k=0}^{\infty} {\phi}_k({0})$, where $\{{\phi}_k\}_{k=0}^{\infty}$ are
the functions in the Veblen hierarchy with finite index. We give a complete
classification of $\mathrm{PredR}_{\mathsf{A}}$ -- for those downsets that
contain at least one infinite ordinal -- in terms of the Grzegorczyk hierarchy
$\{\mathcal{E}_k\}_{k=2}^{\omega}$. In this way, we extend Bellantoni-Cook's
characterization of $\mathcal{E}_2$ (the class of linear-space computable
functions) to obtain a machine-independent and structural characterization of
the entire Grzegorczyk hierarchy.

### 3. [Undirected Multicast Network Coding Gaps via Locally Decodable Codes](http://arxiv.org/pdf/2510.18737v1)

Authors: Mark Braverman, Zhongtian He

The network coding problem asks whether data throughput in a network can be
increased using coding (compared to treating bits as commodities in a flow).
While it is well-known that a network coding advantage exists in directed
graphs, the situation in undirected graphs is much less understood -- in
particular, despite significant effort, it is not even known whether network
coding is helpful at all for unicast sessions.
  In this paper we study the multi-source multicast network coding problem in
undirected graphs. There are $k$ sources broadcasting each to a subset of nodes
in a graph of size $n$. The corresponding combinatorial problem is a version of
the Steiner tree packing problem, and the network coding question asks whether
the multicast coding rate exceeds the tree-packing rate.
  We give the first super-constant bound to this problem, demonstrating an
example with a coding advantage of $\Omega(\log k)$. In terms of graph size, we
obtain a lower bound of $2^{\tilde{\Omega}(\sqrt{\log \log n})}$. We also
obtain an upper bound of $O(\log n)$ on the gap.
  Our main technical contribution is a new reduction that converts
locally-decodable codes in the low-error regime into multicast coding
instances. This gives rise to a new family of explicitly constructed graphs,
which may have other applications.

### Computational Engineering

### 1. [Regional heterogeneity in left atrial stiffness impacts passive deformation in a cohort of patient-specific models](http://arxiv.org/pdf/2510.18642v1)

Authors: Tiffany MG Baptiste, Cristobal Rodero, Charles P Sillett, Marina Strocchi, Christopher W Lanyon, Christoph M Augustin, Angela WC Lee, José Alonso Solís-Lemus, Caroline H Roney, Daniel B Ennis, Ronak Rajani, Christopher A Rinaldi, Gernot Plank, Richard D Wilkinson, Steven E Williams, Steven A Niederer

The deformation of the left atrium (LA), or its biomechanical function, is
closely linked to the health of this cardiac chamber. In atrial fibrillation
(AF), atrial biomechanics are significantly altered but the underlying cause of
this change is not always clear. Patient-specific models of the LA that
replicate patient atrial motion can allow us to understand how factors such as
atrial anatomy, myocardial stiffness and physiological constraints are linked
to atrial biomechanics. We created patient-specific LA models from CT images.
We fitted regional model stiffness to peak CT-derived deformation during the LA
reservoir phase ($\pm0.90$ mm) and used the CT deformation transients through
the reservoir and conduit phase for model validation (deformation transients
fell within $\pm0.38$ mm per unit time of targets). We found that myocardial
stiffness varies regionally across the LA. The regional stiffness values were
significant factors contributing to regional physiological LA deformation
($p=0.023$) while features of LA anatomy, including regional wall thickness and
adipose volume, were less important. These findings provide insight into the
underlying causes of altered LA biomechanics in AF.

### 2. [RAISE: A Unified Framework for Responsible AI Scoring and Evaluation](http://arxiv.org/pdf/2510.18559v1)

Authors: Loc Phuc Truong Nguyen, Hung Thanh Do

As AI systems enter high-stakes domains, evaluation must extend beyond
predictive accuracy to include explainability, fairness, robustness, and
sustainability. We introduce RAISE (Responsible AI Scoring and Evaluation), a
unified framework that quantifies model performance across these four
dimensions and aggregates them into a single, holistic Responsibility Score. We
evaluated three deep learning models: a Multilayer Perceptron (MLP), a Tabular
ResNet, and a Feature Tokenizer Transformer, on structured datasets from
finance, healthcare, and socioeconomics. Our findings reveal critical
trade-offs: the MLP demonstrated strong sustainability and robustness, the
Transformer excelled in explainability and fairness at a very high
environmental cost, and the Tabular ResNet offered a balanced profile. These
results underscore that no single model dominates across all responsibility
criteria, highlighting the necessity of multi-dimensional evaluation for
responsible model selection. Our implementation is available at:
https://github.com/raise-framework/raise.

### Computational Geometry

### 1. [On the Computation of Schrijver's Kernels](http://arxiv.org/pdf/2510.18597v1)

Authors: Vincent Delecroix, Oscar Fontaine, Francis Lazarus

The geometry of a graph $G$ embedded on a closed oriented surface $S$ can be
probed by counting the intersections of $G$ with closed curves on $S$. Of
special interest is the map $c \mapsto \mu_G(c)$ counting the minimum number of
intersections between $G$ and any curve freely homotopic to a given curve $c$.
Schrijver [On the uniqueness of kernels, 1992] calls $G$ a kernel if for any
proper graph minor $H$ of $G$ we have $\mu_H < \mu_G$. Hence, $G$ admits a
minor $H$ which is a kernel and such that $\mu_G = \mu_H$. We show how to
compute such a minor kernel of $G$ in $O(n^3 \log n)$ time where $n$ is the
number of edges of $G$, and $g\ge 2$ is the genus of $S$. Our algorithm
leverages a tight bound on the size of minimal bigons in a system of closed
curves. It also relies on several subroutines of independent interest including
the computation of the area enclosed by a curve and a test of simplicity for
the lift of a curve in the universal covering of $S$.
  As a consequence of our minor kernel algorithm and a recent result of Dubois
[Making multicurves cross minimally on surfaces, 2024], after a preprocessing
that takes $O(n^3 \log n)$ time and $O(n)$ space, we are able to compute
$\mu_G(c)$ in $O(g (n + \ell) \log(n + \ell))$ time given any closed walk $c$
with $\ell$ edges. The state-of-the-art algorithm by Colin de Verdi\`ere and
Erickson [Tightening non-simple paths and cycles on surfaces, 2010] would avoid
constructing a kernel but would lead to a computation of $\mu_G(c)$ in $O(g n
\ell \log(n \ell))$ time (with a preprocessing that takes $O(gn\log n)$ time
and $O(gn)$ space). Another consequence of the computation of minor kernels is
the ability to decide in polynomial time whether two graph minors $H$ and $H'$
of $G$ satisfy $\mu_H = \mu_{H'}$.

### 2. [Bounding the number of holes required for folding rectangular polyomoinoes into cubes](http://arxiv.org/pdf/2510.18197v1)

Authors: Florian Lehner, Benjamin Shirley

We study the problem of whether rectangular polyominoes with holes are
cube-foldable, that is, whether they can be folded into a cube, if creases are
only allowed along grid lines. It is known that holes of sufficient size
guarantee that this is the case.
  Smaller holes which by themselves do not make a rectangular polyomino
cube-foldable can sometimes be combined to create cube-foldable polyominoes. We
investigate minimal sets of holes which guarantee cube-foldability. We show
that if all holes are of the same type, the these minimal sets have size at
most 4, and if we allow different types of holes, then there is no upper bound
on the size.

### Computation and Language

### 1. [MARCUS: An Event-Centric NLP Pipeline that generates Character Arcs from Narratives](http://arxiv.org/pdf/2510.18201v1)

Authors: Sriharsh Bhyravajjula, Ujwal Narayan, Manish Shrivastava

Character arcs are important theoretical devices employed in literary studies
to understand character journeys, identify tropes across literary genres, and
establish similarities between narratives. This work addresses the novel task
of computationally generating event-centric, relation-based character arcs from
narratives. Providing a quantitative representation for arcs brings tangibility
to a theoretical concept and paves the way for subsequent applications. We
present MARCUS (Modelling Arcs for Understanding Stories), an NLP pipeline that
extracts events, participant characters, implied emotion, and sentiment to
model inter-character relations. MARCUS tracks and aggregates these relations
across the narrative to generate character arcs as graphical plots. We generate
character arcs from two extended fantasy series, Harry Potter and Lord of the
Rings. We evaluate our approach before outlining existing challenges,
suggesting applications of our pipeline, and discussing future work.

### 2. [BrailleLLM: Braille Instruction Tuning with Large Language Models for Braille Domain Tasks](http://arxiv.org/pdf/2510.18288v1)

Authors: Tianyuan Huang, Zepeng Zhu, Hangdi Xing, Zirui Shao, Zhi Yu, Chaoxiong Yang, Jiaxian He, Xiaozhong Liu, Jiajun Bu

Braille plays a vital role in education and information accessibility for
visually impaired individuals. However, Braille information processing faces
challenges such as data scarcity and ambiguities in mixed-text contexts. We
construct English and Chinese Braille Mixed Datasets (EBMD/CBMD) with
mathematical formulas to support diverse Braille domain research, and propose a
syntax tree-based augmentation method tailored for Braille data. To address the
underperformance of traditional fine-tuning methods in Braille-related tasks,
we investigate Braille Knowledge-Based Fine-Tuning (BKFT), which reduces the
learning difficulty of Braille contextual features. BrailleLLM employs BKFT via
instruction tuning to achieve unified Braille translation, formula-to-Braille
conversion, and mixed-text translation. Experiments demonstrate that BKFT
achieves significant performance improvements over conventional fine-tuning in
Braille translation scenarios. Our open-sourced datasets and methodologies
establish a foundation for low-resource multilingual Braille research.

### 3. [Combining Distantly Supervised Models with In Context Learning for Monolingual and Cross-Lingual Relation Extraction](http://arxiv.org/pdf/2510.18344v1)

Authors: Vipul Rathore, Malik Hammad Faisal, Parag Singla, Mausam

Distantly Supervised Relation Extraction (DSRE) remains a long-standing
challenge in NLP, where models must learn from noisy bag-level annotations
while making sentence-level predictions. While existing state-of-the-art (SoTA)
DSRE models rely on task-specific training, their integration with in-context
learning (ICL) using large language models (LLMs) remains underexplored. A key
challenge is that the LLM may not learn relation semantics correctly, due to
noisy annotation.
  In response, we propose HYDRE -- HYbrid Distantly Supervised Relation
Extraction framework. It first uses a trained DSRE model to identify the top-k
candidate relations for a given test sentence, then uses a novel dynamic
exemplar retrieval strategy that extracts reliable, sentence-level exemplars
from training data, which are then provided in LLM prompt for outputting the
final relation(s).
  We further extend HYDRE to cross-lingual settings for RE in low-resource
languages. Using available English DSRE training data, we evaluate all methods
on English as well as a newly curated benchmark covering four diverse
low-resource Indic languages -- Oriya, Santali, Manipuri, and Tulu. HYDRE
achieves up to 20 F1 point gains in English and, on average, 17 F1 points on
Indic languages over prior SoTA DSRE models. Detailed ablations exhibit HYDRE's
efficacy compared to other prompting strategies.

### 4. [KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs](http://arxiv.org/pdf/2510.18368v1)

Authors: Donghyeon Ko, Yeguk Jin, Kyubyung Chae, Byungwook Lee, Chansong Jo, Sookyo In, Jaehong Lee, Taesup Kim, Donghyun Kwak

We present $\textbf{Korean SimpleQA (KoSimpleQA)}$, a benchmark for
evaluating factuality in large language models (LLMs) with a focus on Korean
cultural knowledge. KoSimpleQA is designed to be challenging yet easy to grade,
consisting of 1,000 short, fact-seeking questions with unambiguous answers. We
conduct a comprehensive evaluation across a diverse set of open-source LLMs of
varying sizes that support Korean, and find that even the strongest model
generates correct answer only 33.7% of the time, underscoring the challenging
nature of KoSimpleQA. Notably, performance rankings on KoSimpleQA differ
substantially from those on the English SimpleQA, highlighting the unique value
of our dataset. Furthermore, our analysis of reasoning LLMs shows that engaging
reasoning capabilities in the factual QA task can both help models better
elicit their latent knowledge and improve their ability to abstain when
uncertain. KoSimpleQA can be found at
https://anonymous.4open.science/r/KoSimpleQA-62EB.

### 5. [Towards Fair ASR For Second Language Speakers Using Fairness Prompted Finetuning](http://arxiv.org/pdf/2510.18374v1)

Authors: Monorama Swain, Bubai Maji, Jagabandhu Mishra, Markus Schedl, Anders Søgaard, Jesper Rindom Jensen

In this work, we address the challenge of building fair English ASR systems
for second-language speakers. Our analysis of widely used ASR models, Whisper
and Seamless-M4T, reveals large fluctuations in word error rate (WER) across 26
accent groups, indicating significant fairness gaps. To mitigate this, we
propose fairness-prompted finetuning with lightweight adapters, incorporating
Spectral Decoupling (SD), Group Distributionally Robust Optimization
(Group-DRO), and Invariant Risk Minimization (IRM). Our proposed fusion of
traditional empirical risk minimization (ERM) with cross-entropy and
fairness-driven objectives (SD, Group DRO, and IRM) enhances fairness across
accent groups while maintaining overall recognition accuracy. In terms of
macro-averaged word error rate, our approach achieves a relative improvement of
58.7% and 58.5% over the large pretrained Whisper and SeamlessM4T, and 9.7% and
7.8% over them, finetuning with standard empirical risk minimization with
cross-entropy loss.

### 6. [Adamas: Hadamard Sparse Attention for Efficient Long-Context Inference](http://arxiv.org/pdf/2510.18413v1)

Authors: Siyuan Yan, Guo-Qing Jiang, Yuchen Zhang, Xiaoxing Ma, Ran Zhu, Chun Cao, Jingwei Xu

Large language models (LLMs) now support context windows of hundreds of
thousands to millions of tokens, enabling applications such as long-document
summarization, large-scale code synthesis, multi-document question answering
and persistent multi-turn dialogue. However, such extended contexts exacerbate
the quadratic cost of self-attention, leading to severe latency in
autoregressive decoding. Existing sparse attention methods alleviate these
costs but rely on heuristic patterns that struggle to recall critical key-value
(KV) pairs for each query, resulting in accuracy degradation. We introduce
Adamas, a lightweight yet highly accurate sparse attention mechanism designed
for long-context inference. Adamas applies the Hadamard transform,
bucketization and 2-bit compression to produce compact representations, and
leverages Manhattan-distance estimation for efficient top-k selections.
Experiments show that Adamas matches the accuracy of full attention with only a
64-token budget, achieves near-lossless performance at 128, and supports up to
8x higher sparsity than prior state-of-the-art (SOTA) methods while delivering
up to 4.4x self-attention and 1.5x end-to-end speedups on 32K-length sequences.
Remarkably, Adamas attains comparable or even lower perplexity than full
attention, underscoring its effectiveness in maintaining accuracy under
aggressive sparsity.

### 7. [Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response](http://arxiv.org/pdf/2510.18434v1)

Authors: Qingqing Gu, Dan Wang, Yue Zhao, Xiaoyu Wang, Zhonglin Jiang, Yong Chen, Hongyan Li, Luo Ji

Chain-of-Thought (CoT) is widely applied to improve the LLM capability in
math, coding and reasoning tasks. However, its performance is limited for
open-domain tasks since there are no clearly defined reasoning steps or logical
transitions. To mitigate such challenges, we propose another prompt-based
paradigm called Chain of Conceptual Thought (CoCT), where the LLM first tags a
concept, then generates the detailed content. The chain of concepts is allowed
within the utterance, encouraging the LLM's deep and strategic thinking. We
experiment with this paradigm in daily and emotional support conversations
where the concept is comprised of emotions, strategies and topics. Automatic,
human and model evaluations suggest that CoCT surpasses baselines such as
Self-Refine, ECoT, ToT, SoT and RAG, suggesting a potential effective
prompt-based paradigm of LLM for a wider scope of tasks.

### 8. [Grounding or Guessing? Visual Signals for Detecting Hallucinations in Sign Language Translation](http://arxiv.org/pdf/2510.18439v1)

Authors: Yasser Hamidullah, Koel Dutta Chowdury, Yusser Al-Ghussin, Shakib Yazdani, Cennet Oguz, Josef van Genabith, Cristina España-Bonet

Hallucination, where models generate fluent text unsupported by visual
evidence, remains a major flaw in vision-language models and is particularly
critical in sign language translation (SLT). In SLT, meaning depends on precise
grounding in video, and gloss-free models are especially vulnerable because
they map continuous signer movements directly into natural language without
intermediate gloss supervision that serves as alignment. We argue that
hallucinations arise when models rely on language priors rather than visual
input. To capture this, we propose a token-level reliability measure that
quantifies how much the decoder uses visual information. Our method combines
feature-based sensitivity, which measures internal changes when video is
masked, with counterfactual signals, which capture probability differences
between clean and altered video inputs. These signals are aggregated into a
sentence-level reliability score, providing a compact and interpretable measure
of visual grounding. We evaluate the proposed measure on two SLT benchmarks
(PHOENIX-2014T and CSL-Daily) with both gloss-based and gloss-free models. Our
results show that reliability predicts hallucination rates, generalizes across
datasets and architectures, and decreases under visual degradations. Beyond
these quantitative trends, we also find that reliability distinguishes grounded
tokens from guessed ones, allowing risk estimation without references; when
combined with text-based signals (confidence, perplexity, or entropy), it
further improves hallucination risk estimation. Qualitative analysis highlights
why gloss-free models are more susceptible to hallucinations. Taken together,
our findings establish reliability as a practical and reusable tool for
diagnosing hallucinations in SLT, and lay the groundwork for more robust
hallucination detection in multimodal generation.

### 9. [Engagement Undermines Safety: How Stereotypes and Toxicity Shape Humor in Language Models](http://arxiv.org/pdf/2510.18454v1)

Authors: Atharvan Dogra, Soumya Suvra Ghosal, Ameet Deshpande, Ashwin Kalyan, Dinesh Manocha

Large language models are increasingly used for creative writing and
engagement content, raising safety concerns about the outputs. Therefore,
casting humor generation as a testbed, this work evaluates how funniness
optimization in modern LLM pipelines couples with harmful content by jointly
measuring humor, stereotypicality, and toxicity. This is further supplemented
by analyzing incongruity signals through information-theoretic metrics. Across
six models, we observe that harmful outputs receive higher humor scores which
further increase under role-based prompting, indicating a bias amplification
loop between generators and evaluators. Information-theoretic analyses show
harmful cues widen predictive uncertainty and surprisingly, can even make
harmful punchlines more expected for some models, suggesting structural
embedding in learned humor distributions. External validation on an additional
satire-generation task with human perceived funniness judgments shows that LLM
satire increases stereotypicality and typically toxicity, including for closed
models. Quantitatively, stereotypical/toxic jokes gain $10-21\%$ in mean humor
score, stereotypical jokes appear $11\%$ to $28\%$ more often among the jokes
marked funny by LLM-based metric and up to $10\%$ more often in generations
perceived as funny by humans.

### 10. [ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks](http://arxiv.org/pdf/2510.18455v1)

Authors: Liyang He, Yuren Zhang, Ziwei Zhu, Zhenghui Li, Shiwei Tong

Retrieval Augmented Generation (RAG) systems are increasingly vital in
dynamic domains like online gaming, yet the lack of a dedicated benchmark has
impeded standardized evaluation in this area. The core difficulty lies in Dual
Dynamics: the constant interplay between game content updates and the shifting
focus of the player community. Furthermore, the necessity of automating such a
benchmark introduces a critical requirement for player-centric authenticity to
ensure generated questions are realistic. To address this integrated challenge,
we introduce ChronoPlay, a novel framework for the automated and continuous
generation of game RAG benchmarks. ChronoPlay utilizes a dual-dynamic update
mechanism to track both forms of change, and a dual-source synthesis engine
that draws from official sources and player community to ensure both factual
correctness and authentic query patterns. We instantiate our framework on three
distinct games to create the first dynamic RAG benchmark for the gaming domain,
offering new insights into model performance under these complex and realistic
conditions. Code is avaliable at: https://github.com/hly1998/ChronoPlay.

### Cryptography and Security

### 1. [TaintSentinel: Path-Level Randomness Vulnerability Detection for Ethereum Smart Contracts](http://arxiv.org/pdf/2510.18192v1)

Authors: Hadis Rezaei, Ahmed Afif Monrat, Karl Andersson, Francesco Flammini

The inherent determinism of blockchain technology poses a significant
challenge to generating secure random numbers within smart contracts, leading
to exploitable vulnerabilities, particularly in decentralized finance (DeFi)
ecosystems and blockchain-based gaming applications. From our observations, the
current state-of-the-art detection tools suffer from inadequate precision while
dealing with random number vulnerabilities. To address this problem, we propose
TaintSentinel, a novel path sensitive vulnerability detection system designed
to analyze smart contracts at the execution path level and gradually analyze
taint with domain-specific rules. This paper discusses a solution that
incorporates a multi-faceted approach, integrating rule-based taint analysis to
track data flow, a dual stream neural network to identify complex vulnerability
signatures, and evidence-based parameter initialization to minimize false
positives. The system's two-phase operation involves semantic graph
construction and taint propagation analysis, followed by pattern recognition
using PathGNN and global structural analysis via GlobalGCN. Our experiments on
4,844 contracts demonstrate the superior performance of TaintSentinel relative
to existing tools, yielding an F1-score of 0.892, an AUC-ROC of 0.94, and a PRA
accuracy of 97%.

### 2. [CryptoGuard: Lightweight Hybrid Detection and Response to Host-based Cryptojackers in Linux Cloud Environments](http://arxiv.org/pdf/2510.18324v1)

Authors: Gyeonghoon Park, Jaehan Kim, Jinu Choi, Jinwoo Kim

Host-based cryptomining malware, commonly known as cryptojackers, have gained
notoriety for their stealth and the significant financial losses they cause in
Linux-based cloud environments. Existing solutions often struggle with
scalability due to high monitoring overhead, low detection accuracy against
obfuscated behavior, and lack of integrated remediation. We present
CryptoGuard, a lightweight hybrid solution that combines detection and
remediation strategies to counter cryptojackers. To ensure scalability,
CryptoGuard uses sketch- and sliding window-based syscall monitoring to collect
behavior patterns with minimal overhead. It decomposes the classification task
into a two-phase process, leveraging deep learning models to identify
suspicious activity with high precision. To counter evasion techniques such as
entry point poisoning and PID manipulation, CryptoGuard integrates targeted
remediation mechanisms based on eBPF, a modern Linux kernel feature deployable
on any compatible host. Evaluated on 123 real-world cryptojacker samples, it
achieves average F1-scores of 96.12% and 92.26% across the two phases, and
outperforms state-of-the-art baselines in terms of true and false positive
rates, while incurring only 0.06% CPU overhead per host.

### 3. [DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning](http://arxiv.org/pdf/2510.18438v1)

Authors: Yixuan Liu, Xinlei Li, Yi Li

Phishing attacks in Web3 ecosystems are increasingly sophisticated,
exploiting deceptive contract logic, malicious frontend scripts, and token
approval patterns. We present DeepTx, a real-time transaction analysis system
that detects such threats before user confirmation. DeepTx simulates pending
transactions, extracts behavior, context, and UI features, and uses multiple
large language models (LLMs) to reason about transaction intent. A consensus
mechanism with self-reflection ensures robust and explainable decisions.
Evaluated on our phishing dataset, DeepTx achieves high precision and recall
(demo video: https://youtu.be/4OfK9KCEXUM).

### 4. [PP3D: An In-Browser Vision-Based Defense Against Web Behavior Manipulation Attacks](http://arxiv.org/pdf/2510.18465v1)

Authors: Spencer King, Irfan Ozen, Karthika Subramani, Saranyan Senthivel, Phani Vadrevu, Roberto Perdisci

Web-based behavior-manipulation attacks (BMAs) - such as scareware, fake
software downloads, tech support scams, etc. - are a class of social
engineering (SE) attacks that exploit human decision-making vulnerabilities.
These attacks remain under-studied compared to other attacks such as
information harvesting attacks (e.g., phishing) or malware infections. Prior
technical work has primarily focused on measuring BMAs, offering little in the
way of generic defenses.
  To address this gap, we introduce Pixel Patrol 3D (PP3D), the first
end-to-end browser framework for discovering, detecting, and defending against
behavior-manipulating SE attacks in real time. PP3D consists of a visual
detection model implemented within a browser extension, which deploys the model
client-side to protect users across desktop and mobile devices while preserving
privacy.
  Our evaluation shows that PP3D can achieve above 99% detection rate at 1%
false positives, while maintaining good latency and overhead performance across
devices. Even when faced with new BMA samples collected months after training
the detection model, our defense system can still achieve above 97% detection
rate at 1% false positives. These results demonstrate that our framework offers
a practical, effective, and generalizable defense against a broad and evolving
class of web behavior-manipulation attacks.

### 5. [The Attribution Story of WhisperGate: An Academic Perspective](http://arxiv.org/pdf/2510.18484v1)

Authors: Oleksandr Adamov, Anders Carlsson

This paper explores the challenges of cyberattack attribution, specifically
APTs, applying the case study approach for the WhisperGate cyber operation of
January 2022 executed by the Russian military intelligence service (GRU) and
targeting Ukrainian government entities. The study provides a detailed review
of the threat actor identifiers and taxonomies used by leading cybersecurity
vendors, focusing on the evolving attribution from Microsoft, ESET, and
CrowdStrike researchers. Once the attribution to Ember Bear (GRU Unit 29155) is
established through technical and intelligence reports, we use both traditional
machine learning classifiers and a large language model (ChatGPT) to analyze
the indicators of compromise (IoCs), tactics, and techniques to statistically
and semantically attribute the WhisperGate attack. Our findings reveal
overlapping indicators with the Sandworm group (GRU Unit 74455) but also strong
evidence pointing to Ember Bear, especially when the LLM is fine-tuned or
contextually augmented with additional intelligence. Thus, showing how AI/GenAI
with proper fine-tuning are capable of solving the attribution challenge.

### 6. [Prompting the Priorities: A First Look at Evaluating LLMs for Vulnerability Triage and Prioritization](http://arxiv.org/pdf/2510.18508v1)

Authors: Osama Al Haddad, Muhammad Ikram, Ejaz Ahmed, Young Lee

Security analysts face increasing pressure to triage large and complex
vulnerability backlogs. Large Language Models (LLMs) offer a potential aid by
automating parts of the interpretation process. We evaluate four models
(ChatGPT, Claude, Gemini, and DeepSeek) across twelve prompting techniques to
interpret semi-structured and unstructured vulnerability information. As a
concrete use case, we test each model's ability to predict decision points in
the Stakeholder-Specific Vulnerability Categorization (SSVC) framework:
Exploitation, Automatable, Technical Impact, and Mission and Wellbeing.
  Using 384 real-world vulnerabilities from the VulZoo dataset, we issued more
than 165,000 queries to assess performance under prompting styles including
one-shot, few-shot, and chain-of-thought. We report F1 scores for each SSVC
decision point and Cohen's kappa (weighted and unweighted) for the final SSVC
decision outcomes. Gemini consistently ranked highest, leading on three of four
decision points and yielding the most correct recommendations. Prompting with
exemplars generally improved accuracy, although all models struggled on some
decision points. Only DeepSeek achieved fair agreement under weighted metrics,
and all models tended to over-predict risk.
  Overall, current LLMs do not replace expert judgment. However, specific LLM
and prompt combinations show moderate effectiveness for targeted SSVC
decisions. When applied with care, LLMs can support vulnerability
prioritization workflows and help security teams respond more efficiently to
emerging threats.

### 7. [Deep Q-Learning Assisted Bandwidth Reservation for Multi-Operator Time-Sensitive Vehicular Networking](http://arxiv.org/pdf/2510.18553v1)

Authors: Abdullah Al-Khatib, Albert Gergus, Muneeb Ul Hassan, Abdelmajid Khelil, Klaus Mossner, Holger Timinger

Very few available individual bandwidth reservation schemes provide efficient
and cost-effective bandwidth reservation that is required for safety-critical
and time-sensitive vehicular networked applications. These schemes allow
vehicles to make reservation requests for the required resources. Accordingly,
a Mobile Network Operator (MNO) can allocate and guarantee bandwidth resources
based on these requests. However, due to uncertainty in future reservation time
and bandwidth costs, the design of an optimized reservation strategy is
challenging. In this article, we propose a novel multi-objective bandwidth
reservation update approach with an optimal strategy based on Double Deep
Q-Network (DDQN). The key design objectives are to minimize the reservation
cost with multiple MNOs and to ensure reliable resource provisioning in
uncertain situations by solving scenarios such as underbooked and overbooked
reservations along the driving path. The enhancements and advantages of our
proposed strategy have been demonstrated through extensive experimental results
when compared to other methods like greedy update or other deep reinforcement
learning approaches. Our strategy demonstrates a 40% reduction in bandwidth
costs across all investigated scenarios and simultaneously resolves uncertain
situations in a cost-effective manner.

### 8. [The Trust Paradox in LLM-Based Multi-Agent Systems: When Collaboration Becomes a Security Vulnerability](http://arxiv.org/pdf/2510.18563v1)

Authors: Zijie Xu, Minfeng Qi, Shiqing Wu, Lefeng Zhang, Qiwen Wei, Han He, Ningran Li

Multi-agent systems powered by large language models are advancing rapidly,
yet the tension between mutual trust and security remains underexplored. We
introduce and empirically validate the Trust-Vulnerability Paradox (TVP):
increasing inter-agent trust to enhance coordination simultaneously expands
risks of over-exposure and over-authorization. To investigate this paradox, we
construct a scenario-game dataset spanning 3 macro scenes and 19 sub-scenes,
and run extensive closed-loop interactions with trust explicitly parameterized.
Using Minimum Necessary Information (MNI) as the safety baseline, we propose
two unified metrics: Over-Exposure Rate (OER) to detect boundary violations,
and Authorization Drift (AD) to capture sensitivity to trust levels. Results
across multiple model backends and orchestration frameworks reveal consistent
trends: higher trust improves task success but also heightens exposure risks,
with heterogeneous trust-to-risk mappings across systems. We further examine
defenses such as Sensitive Information Repartitioning and Guardian-Agent
enablement, both of which reduce OER and attenuate AD. Overall, this study
formalizes TVP, establishes reproducible baselines with unified metrics, and
demonstrates that trust must be modeled and scheduled as a first-class security
variable in multi-agent system design.

### 9. [Privacy-Preserving Healthcare Data in IoT: A Synergistic Approach with Deep Learning and Blockchain](http://arxiv.org/pdf/2510.18568v1)

Authors: Behnam Rezaei Bezanjani, Seyyed Hamid Ghafouri, Reza Gholamrezaei

The integration of Internet of Things (IoT) devices in healthcare has
revolutionized patient care by enabling real-time monitoring, personalized
treatments, and efficient data management. However, this technological
advancement introduces significant security risks, particularly concerning the
confidentiality, integrity, and availability of sensitive medical data.
Traditional security measures are often insufficient to address the unique
challenges posed by IoT environments, such as heterogeneity, resource
constraints, and the need for real-time processing. To tackle these challenges,
we propose a comprehensive three-phase security framework designed to enhance
the security and reliability of IoT-enabled healthcare systems. In the first
phase, the framework assesses the reliability of IoT devices using a
reputation-based trust estimation mechanism, which combines device behavior
analytics with off-chain data storage to ensure scalability. The second phase
integrates blockchain technology with a lightweight proof-of-work mechanism,
ensuring data immutability, secure communication, and resistance to
unauthorized access. The third phase employs a lightweight Long Short-Term
Memory (LSTM) model for anomaly detection and classification, enabling
real-time identification of cyber threats. Simulation results demonstrate that
the proposed framework outperforms existing methods, achieving a 2% increase in
precision, accuracy, and recall, a 5% higher attack detection rate, and a 3%
reduction in false alarm rate. These improvements highlight the framework's
ability to address critical security concerns while maintaining scalability and
real-time performance.

### 10. [CLASP: Cost-Optimized LLM-based Agentic System for Phishing Detection](http://arxiv.org/pdf/2510.18585v1)

Authors: Fouad Trad, Ali Chehab

Phishing websites remain a significant cybersecurity threat, necessitating
accurate and cost-effective detection mechanisms. In this paper, we present
CLASP, a novel system that effectively identifies phishing websites by
leveraging multiple intelligent agents, built using large language models
(LLMs), to analyze different aspects of a web resource. The system processes
URLs or QR codes, employing specialized LLM-based agents that evaluate the URL
structure, webpage screenshot, and HTML content to predict potential phishing
threats. To optimize performance while minimizing operational costs, we
experimented with multiple combination strategies for agent-based analysis,
ultimately designing a strategic combination that ensures the per-website
evaluation expense remains minimal without compromising detection accuracy. We
tested various LLMs, including Gemini 1.5 Flash and GPT-4o mini, to build these
agents and found that Gemini 1.5 Flash achieved the best performance with an F1
score of 83.01% on a newly curated dataset. Also, the system maintained an
average processing time of 2.78 seconds per website and an API cost of around
$3.18 per 1,000 websites. Moreover, CLASP surpasses leading previous solutions,
achieving over 40% higher recall and a 20% improvement in F1 score for phishing
detection on the collected dataset. To support further research, we have made
our dataset publicly available, supporting the development of more advanced
phishing detection systems.

### Computer Vision and Pattern Recognition

### 1. [EMA-SAM: Exponential Moving-average for SAM-based PTMC Segmentation](http://arxiv.org/pdf/2510.18213v1)

Authors: Maryam Dialameh, Hossein Rajabzadeh, Jung Suk Sim, Hyock Ju Kwon

Papillary thyroid microcarcinoma (PTMC) is increasingly managed with
radio-frequency ablation (RFA), yet accurate lesion segmentation in ultrasound
videos remains difficult due to low contrast, probe-induced motion, and
heat-related artifacts. The recent Segment Anything Model 2 (SAM-2) generalizes
well to static images, but its frame-independent design yields unstable
predictions and temporal drift in interventional ultrasound. We introduce
\textbf{EMA-SAM}, a lightweight extension of SAM-2 that incorporates a
confidence-weighted exponential moving average pointer into the memory bank,
providing a stable latent prototype of the tumour across frames. This design
preserves temporal coherence through probe pressure and bubble occlusion while
rapidly adapting once clear evidence reappears. On our curated PTMC-RFA dataset
(124 minutes, 13 patients), EMA-SAM improves \emph{maxDice} from 0.82 (SAM-2)
to 0.86 and \emph{maxIoU} from 0.72 to 0.76, while reducing false positives by
29\%. On external benchmarks, including VTUS and colonoscopy video polyp
datasets, EMA-SAM achieves consistent gains of 2--5 Dice points over SAM-2.
Importantly, the EMA pointer adds \textless0.1\% FLOPs, preserving real-time
throughput of $\sim$30\,FPS on a single A100 GPU. These results establish
EMA-SAM as a robust and efficient framework for stable tumour tracking,
bridging the gap between foundation models and the stringent demands of
interventional ultrasound. Codes are available here \hyperref[code
{https://github.com/mdialameh/EMA-SAM}.

### 2. [Beyond Frequency: Scoring-Driven Debiasing for Object Detection via Blueprint-Prompted Image Synthesis](http://arxiv.org/pdf/2510.18229v1)

Authors: Xinhao Cai, Liulei Li, Gensheng Pei, Tao Chen, Jinshan Pan, Yazhou Yao, Wenguan Wang

This paper presents a generation-based debiasing framework for object
detection. Prior debiasing methods are often limited by the representation
diversity of samples, while naive generative augmentation often preserves the
biases it aims to solve. Moreover, our analysis reveals that simply generating
more data for rare classes is suboptimal due to two core issues: i) instance
frequency is an incomplete proxy for the true data needs of a model, and ii)
current layout-to-image synthesis lacks the fidelity and control to generate
high-quality, complex scenes. To overcome this, we introduce the representation
score (RS) to diagnose representational gaps beyond mere frequency, guiding the
creation of new, unbiased layouts. To ensure high-quality synthesis, we replace
ambiguous text prompts with a precise visual blueprint and employ a generative
alignment strategy, which fosters communication between the detector and
generator. Our method significantly narrows the performance gap for
underrepresented object groups, \eg, improving large/rare instances by 4.4/3.6
mAP over the baseline, and surpassing prior L2I synthesis models by 15.9 mAP
for layout accuracy in generated images.

### 3. [DeepSeek-OCR: Contexts Optical Compression](http://arxiv.org/pdf/2510.18234v1)

Authors: Haoran Wei, Yaofeng Sun, Yukun Li

We present DeepSeek-OCR as an initial investigation into the feasibility of
compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two
components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically,
DeepEncoder serves as the core engine, designed to maintain low activations
under high-resolution input while achieving high compression ratios to ensure
an optimal and manageable number of vision tokens. Experiments show that when
the number of text tokens is within 10 times that of vision tokens (i.e., a
compression ratio < 10x), the model can achieve decoding (OCR) precision of
97%. Even at a compression ratio of 20x, the OCR accuracy still remains at
about 60%. This shows considerable promise for research areas such as
historical long-context compression and memory forgetting mechanisms in LLMs.
Beyond this, DeepSeek-OCR also demonstrates high practical value. On
OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision
tokens, and outperforms MinerU2.0 (6000+ tokens per page on average) while
utilizing fewer than 800 vision tokens. In production, DeepSeek-OCR can
generate training data for LLMs/VLMs at a scale of 200k+ pages per day (a
single A100-40G). Codes and model weights are publicly accessible at
http://github.com/deepseek-ai/DeepSeek-OCR.

### 4. [BlendCLIP: Bridging Synthetic and Real Domains for Zero-Shot 3D Object Classification with Multimodal Pretraining](http://arxiv.org/pdf/2510.18244v1)

Authors: Ajinkya Khoche, Gergő László Nagy, Maciej Wozniak, Thomas Gustafsson, Patric Jensfelt

Zero-shot 3D object classification is crucial for real-world applications
like autonomous driving, however it is often hindered by a significant domain
gap between the synthetic data used for training and the sparse, noisy LiDAR
scans encountered in the real-world. Current methods trained solely on
synthetic data fail to generalize to outdoor scenes, while those trained only
on real data lack the semantic diversity to recognize rare or unseen objects.
  We introduce BlendCLIP, a multimodal pretraining framework that bridges this
synthetic-to-real gap by strategically combining the strengths of both domains.
We first propose a pipeline to generate a large-scale dataset of object-level
triplets -- consisting of a point cloud, image, and text description -- mined
directly from real-world driving data and human annotated 3D boxes. Our core
contribution is a curriculum-based data mixing strategy that first grounds the
model in the semantically rich synthetic CAD data before progressively adapting
it to the specific characteristics of real-world scans.
  Our experiments show that our approach is highly label-efficient: introducing
as few as 1.5\% real-world samples per batch into training boosts zero-shot
accuracy on the nuScenes benchmark by 27\%. Consequently, our final model
achieves state-of-the-art performance on challenging outdoor datasets like
nuScenes and TruckScenes, improving over the best prior method by 19.3\% on
nuScenes, while maintaining strong generalization on diverse synthetic
benchmarks. Our findings demonstrate that effective domain adaptation, not
full-scale real-world annotation, is the key to unlocking robust
open-vocabulary 3D perception. Our code and dataset will be released upon
acceptance on https://github.com/kesu1/BlendCLIP.

### 5. [OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion](http://arxiv.org/pdf/2510.18253v1)

Authors: Tianyu Huang, Runnan Chen, Dongting Hu, Fengming Huang, Mingming Gong, Tongliang Liu

Understanding 3D scenes is pivotal for autonomous driving, robotics, and
augmented reality. Recent semantic Gaussian Splatting approaches leverage
large-scale 2D vision models to project 2D semantic features onto 3D scenes.
However, they suffer from two major limitations: (1) insufficient contextual
cues for individual masks during preprocessing and (2) inconsistencies and
missing details when fusing multi-view features from these 2D models. In this
paper, we introduce \textbf{OpenInsGaussian}, an \textbf{Open}-vocabulary
\textbf{Ins}tance \textbf{Gaussian} segmentation framework with Context-aware
Cross-view Fusion. Our method consists of two modules: Context-Aware Feature
Extraction, which augments each mask with rich semantic context, and
Attention-Driven Feature Aggregation, which selectively fuses multi-view
features to mitigate alignment errors and incompleteness. Through extensive
experiments on benchmark datasets, OpenInsGaussian achieves state-of-the-art
results in open-vocabulary 3D Gaussian segmentation, outperforming existing
baselines by a large margin. These findings underscore the robustness and
generality of our proposed approach, marking a significant step forward in 3D
scene understanding and its practical deployment across diverse real-world
scenarios.

### 6. [UWBench: A Comprehensive Vision-Language Benchmark for Underwater Understanding](http://arxiv.org/pdf/2510.18262v1)

Authors: Da Zhang, Chenggang Rong, Bingyu Li, Feiyu Wang, Zhiyuan Zhao, Junyu Gao, Xuelong Li

Large vision-language models (VLMs) have achieved remarkable success in
natural scene understanding, yet their application to underwater environments
remains largely unexplored. Underwater imagery presents unique challenges
including severe light attenuation, color distortion, and suspended particle
scattering, while requiring specialized knowledge of marine ecosystems and
organism taxonomy. To bridge this gap, we introduce UWBench, a comprehensive
benchmark specifically designed for underwater vision-language understanding.
UWBench comprises 15,003 high-resolution underwater images captured across
diverse aquatic environments, encompassing oceans, coral reefs, and deep-sea
habitats. Each image is enriched with human-verified annotations including
15,281 object referring expressions that precisely describe marine organisms
and underwater structures, and 124,983 question-answer pairs covering diverse
reasoning capabilities from object recognition to ecological relationship
understanding. The dataset captures rich variations in visibility, lighting
conditions, and water turbidity, providing a realistic testbed for model
evaluation. Based on UWBench, we establish three comprehensive benchmarks:
detailed image captioning for generating ecologically informed scene
descriptions, visual grounding for precise localization of marine organisms,
and visual question answering for multimodal reasoning about underwater
environments. Extensive experiments on state-of-the-art VLMs demonstrate that
underwater understanding remains challenging, with substantial room for
improvement. Our benchmark provides essential resources for advancing
vision-language research in underwater contexts and supporting applications in
marine science, ecological monitoring, and autonomous underwater exploration.
Our code and benchmark will be available.

### 7. [TreeFedDG: Alleviating Global Drift in Federated Domain Generalization for Medical Image Segmentation](http://arxiv.org/pdf/2510.18268v1)

Authors: Yucheng Song, Chenxi Li, Haokang Ding, Zhining Liao, Zhifang Liao

In medical image segmentation tasks, Domain Generalization (DG) under the
Federated Learning (FL) framework is crucial for addressing challenges related
to privacy protection and data heterogeneity. However, traditional federated
learning methods fail to account for the imbalance in information aggregation
across clients in cross-domain scenarios, leading to the Global Drift (GD)
problem and a consequent decline in model generalization performance. This
motivates us to delve deeper and define a new critical issue: global drift in
federated domain generalization for medical imaging (FedDG-GD). In this paper,
we propose a novel tree topology framework called TreeFedDG. First, starting
from the distributed characteristics of medical images, we design a
hierarchical parameter aggregation method based on a tree-structured topology
to suppress deviations in the global model direction. Second, we introduce a
parameter difference-based style mixing method (FedStyle), which enforces
mixing among clients with maximum parameter differences to enhance robustness
against drift. Third, we develop a a progressive personalized fusion strategy
during model distribution, ensuring a balance between knowledge transfer and
personalized features. Finally, during the inference phase, we use feature
similarity to guide the retrieval of the most relevant model chain from the
tree structure for ensemble decision-making, thereby fully leveraging the
advantages of hierarchical knowledge. We conducted extensive experiments on two
publicly available datasets. The results demonstrate that our method
outperforms other state-of-the-art domain generalization approaches in these
challenging tasks and achieves better balance in cross-domain performance.

### 8. [GeoDiff: Geometry-Guided Diffusion for Metric Depth Estimation](http://arxiv.org/pdf/2510.18291v1)

Authors: Tuan Pham, Thanh-Tung Le, Xiaohui Xie, Stephan Mandt

We introduce a novel framework for metric depth estimation that enhances
pretrained diffusion-based monocular depth estimation (DB-MDE) models with
stereo vision guidance. While existing DB-MDE methods excel at predicting
relative depth, estimating absolute metric depth remains challenging due to
scale ambiguities in single-image scenarios. To address this, we reframe depth
estimation as an inverse problem, leveraging pretrained latent diffusion models
(LDMs) conditioned on RGB images, combined with stereo-based geometric
constraints, to learn scale and shift for accurate depth recovery. Our
training-free solution seamlessly integrates into existing DB-MDE frameworks
and generalizes across indoor, outdoor, and complex environments. Extensive
experiments demonstrate that our approach matches or surpasses state-of-the-art
methods, particularly in challenging scenarios involving translucent and
specular surfaces, all without requiring retraining.

### 9. [Proactive Reasoning-with-Retrieval Framework for Medical Multimodal Large Language Models](http://arxiv.org/pdf/2510.18303v1)

Authors: Lehan Wang, Yi Qin, Honglong Yang, Xiaomeng Li

Incentivizing the reasoning ability of Multimodal Large Language Models
(MLLMs) is essential for medical applications to transparently analyze medical
scans and provide reliable diagnosis. However, existing medical MLLMs rely
solely on internal knowledge during reasoning, leading to hallucinated
reasoning and factual inaccuracies when encountering cases beyond their
training scope. Although recent Agentic Retrieval-Augmented Generation (RAG)
methods elicit the medical model's proactive retrieval ability during
reasoning, they are confined to unimodal LLMs, neglecting the crucial visual
information during reasoning and retrieval. Consequently, we propose the first
Multimodal Medical Reasoning-with-Retrieval framework, Med-RwR, which actively
retrieves external knowledge by querying observed symptoms or domain-specific
medical concepts during reasoning. Specifically, we design a two-stage
reinforcement learning strategy with tailored rewards that stimulate the model
to leverage both visual diagnostic findings and textual clinical information
for effective retrieval. Building on this foundation, we further propose a
Confidence-Driven Image Re-retrieval (CDIR) method for test-time scaling when
low prediction confidence is detected. Evaluation on various public medical
benchmarks demonstrates Med-RwR's significant improvements over baseline
models, proving the effectiveness of enhancing reasoning capabilities with
external knowledge integration. Furthermore, Med-RwR demonstrates remarkable
generalizability to unfamiliar domains, evidenced by 8.8% performance gain on
our proposed EchoCardiography Benchmark (ECBench), despite the scarcity of
echocardiography data in the training corpus. Our data, model, and codes will
be made publicly available at https://github.com/xmed-lab/Med-RwR.

### 10. [OmniNWM: Omniscient Driving Navigation World Models](http://arxiv.org/pdf/2510.18313v1)

Authors: Bohan Li, Zhuang Ma, Dalong Du, Baorui Peng, Zhujin Liang, Zhenqiang Liu, Chao Ma, Yueming Jin, Hao Zhao, Wenjun Zeng, Xin Jin

Autonomous driving world models are expected to work effectively across three
core dimensions: state, action, and reward. Existing models, however, are
typically restricted to limited state modalities, short video sequences,
imprecise action control, and a lack of reward awareness. In this paper, we
introduce OmniNWM, an omniscient panoramic navigation world model that
addresses all three dimensions within a unified framework. For state, OmniNWM
jointly generates panoramic videos of RGB, semantics, metric depth, and 3D
occupancy. A flexible forcing strategy enables high-quality long-horizon
auto-regressive generation. For action, we introduce a normalized panoramic
Plucker ray-map representation that encodes input trajectories into pixel-level
signals, enabling highly precise and generalizable control over panoramic video
generation. Regarding reward, we move beyond learning reward functions with
external image-based models: instead, we leverage the generated 3D occupancy to
directly define rule-based dense rewards for driving compliance and safety.
Extensive experiments demonstrate that OmniNWM achieves state-of-the-art
performance in video generation, control accuracy, and long-horizon stability,
while providing a reliable closed-loop evaluation framework through
occupancy-grounded rewards. Project page is available at
https://github.com/Arlo0o/OmniNWM.

### Computers and Society

### 1. [Integrating Large Language Models and Evaluating Student Outcomes in an Introductory Computer Science Course](http://arxiv.org/pdf/2510.18806v1)

Authors: Annapurna Vadaparty, David H. Smith IV, Samvrit Srinath, Mounika Padala, Christine Alvarado, Jamie Gorson Benario, Daniel Zingaro, Leo Porter

Generative AI (GenAI) models have broad implications for education in
general, impacting the foundations of what we teach and how we assess. This is
especially true in computing, where LLMs tuned for coding have demonstrated
shockingly good performance on the types of assignments historically used in
introductory CS (CS1) courses. As a result, CS1 courses will need to change
what skills are taught and how they are assessed. Computing education
researchers have begun to study student use of LLMs, but there remains much to
be understood about the ways that these tools affect student outcomes. In this
paper, we present the design and evaluation of a new CS1 course at a large
research-intensive university that integrates the use of LLMs as a learning
tool for students. We describe the design principles used to create our new
CS1-LLM course, our new course objectives, and evaluation of student outcomes
and perceptions throughout the course as measured by assessment scores and
surveys. Our findings suggest that 1) student exam performance outcomes,
including differences among demographic groups, are largely similar to
historical outcomes for courses without integration of LLM tools, 2) large,
open-ended projects may be particularly valuable in an LLM context, and 3)
students predominantly found the LLM tools helpful, although some had concerns
regarding over-reliance on the tools.

### 2. [Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](http://arxiv.org/pdf/2510.18238v1)

Authors: Bryan Wilder, Angela Zhou

There has been increasing research interest in AI/ML for social impact, and
correspondingly more publication venues have refined review criteria for
practice-driven AI/ML research. However, these review guidelines tend to most
concretely recognize projects that simultaneously achieve deployment and novel
ML methodological innovation. We argue that this introduces incentives for
researchers that undermine the sustainability of a broader research ecosystem
of social impact, which benefits from projects that make contributions on
single front (applied or methodological) that may better meet project partner
needs. Our position is that researchers and reviewers in machine learning for
social impact must simultaneously adopt: 1) a more expansive conception of
social impacts beyond deployment and 2) more rigorous evaluations of the impact
of deployed systems.

### 3. [The Cost-Benefit of Interdisciplinarity in AI for Mental Health](http://arxiv.org/pdf/2510.18581v1)

Authors: Katerina Drakos, Eva Paraschou, Simay Toplu, Line Harder Clemmensen, Christoph Lütge, Nicole Nadine Lønfeldt, Sneha Das

Artificial intelligence has been introduced as a way to improve access to
mental health support. However, most AI mental health chatbots rely on a
limited range of disciplinary input, and fail to integrate expertise across the
chatbot's lifecycle. This paper examines the cost-benefit trade-off of
interdisciplinary collaboration in AI mental health chatbots. We argue that
involving experts from technology, healthcare, ethics, and law across key
lifecycle phases is essential to ensure value-alignment and compliance with the
high-risk requirements of the AI Act. We also highlight practical
recommendations and existing frameworks to help balance the challenges and
benefits of interdisciplinarity in mental health chatbots.

### 4. [MoveOD: Synthesizing Origin-Destination Commute Distribution from U.S. Census Data](http://arxiv.org/pdf/2510.18858v1)

Authors: Rishav Sen, Abhishek Dubey, Ayan Mukhopadhyay, Samitha Samaranayake, Aron Laszka

High-resolution origin-destination (OD) tables are essential for a wide
spectrum of transportation applications, from modeling traffic and signal
timing optimization to congestion pricing and vehicle routing. However, outside
a handful of data rich cities, such data is rarely available. We introduce
MOVEOD, an open-source pipeline that synthesizes public data into commuter OD
flows with fine-grained spatial and temporal departure times for any county in
the United States. MOVEOD combines five open data sources: American Community
Survey (ACS) departure time and travel time distributions, Longitudinal
Employer-Household Dynamics (LODES) residence-to-workplace flows, county
geometries, road network information from OpenStreetMap (OSM), and building
footprints from OSM and Microsoft, into a single OD dataset. We use a
constrained sampling and integer-programming method to reconcile the OD dataset
with data from ACS and LODES. Our approach involves: (1) matching commuter
totals per origin zone, (2) aligning workplace destinations with employment
distributions, and (3) calibrating travel durations to ACS-reported commute
times. This ensures the OD data accurately reflects commuting patterns. We
demonstrate the framework on Hamilton County, Tennessee, where we generate
roughly 150,000 synthetic trips in minutes, which we feed into a benchmark
suite of classical and learning-based vehicle-routing algorithms. The MOVEOD
pipeline is an end-to-end automated system, enabling users to easily apply it
across the United States by giving only a county and a year; and it can be
adapted to other countries with comparable census datasets. The source code and
a lightweight browser interface are publicly available.

### 5. [Food4All: A Multi-Agent Framework for Real-time Free Food Discovery with Integrated Nutritional Metadata](http://arxiv.org/pdf/2510.18289v1)

Authors: Zhengqing Yuan, Yiyang Li, Weixiang Sun, Zheyuan Zhang, Kaiwen Shi, Keerthiram Murugesan, Yanfang Ye

Food insecurity remains a persistent public health emergency in the United
States, tightly interwoven with chronic disease, mental illness, and opioid
misuse. Yet despite the existence of thousands of food banks and pantries,
access remains fragmented: 1) current retrieval systems depend on static
directories or generic search engines, which provide incomplete and
geographically irrelevant results; 2) LLM-based chatbots offer only vague
nutritional suggestions and fail to adapt to real-world constraints such as
time, mobility, and transportation; and 3) existing food recommendation systems
optimize for culinary diversity but overlook survival-critical needs of
food-insecure populations, including immediate proximity, verified
availability, and contextual barriers. These limitations risk leaving the most
vulnerable individuals, those experiencing homelessness, addiction, or digital
illiteracy, unable to access urgently needed resources. To address this, we
introduce Food4All, the first multi-agent framework explicitly designed for
real-time, context-aware free food retrieval. Food4All unifies three
innovations: 1) heterogeneous data aggregation across official databases,
community platforms, and social media to provide a continuously updated pool of
food resources; 2) a lightweight reinforcement learning algorithm trained on
curated cases to optimize for both geographic accessibility and nutritional
correctness; and 3) an online feedback loop that dynamically adapts retrieval
policies to evolving user needs. By bridging information acquisition, semantic
analysis, and decision support, Food4All delivers nutritionally annotated and
guidance at the point of need. This framework establishes an urgent step toward
scalable, equitable, and intelligent systems that directly support populations
facing food insecurity and its compounding health risks.

### 6. [RAISE: A Unified Framework for Responsible AI Scoring and Evaluation](http://arxiv.org/pdf/2510.18559v1)

Authors: Loc Phuc Truong Nguyen, Hung Thanh Do

As AI systems enter high-stakes domains, evaluation must extend beyond
predictive accuracy to include explainability, fairness, robustness, and
sustainability. We introduce RAISE (Responsible AI Scoring and Evaluation), a
unified framework that quantifies model performance across these four
dimensions and aggregates them into a single, holistic Responsibility Score. We
evaluated three deep learning models: a Multilayer Perceptron (MLP), a Tabular
ResNet, and a Feature Tokenizer Transformer, on structured datasets from
finance, healthcare, and socioeconomics. Our findings reveal critical
trade-offs: the MLP demonstrated strong sustainability and robustness, the
Transformer excelled in explainability and fairness at a very high
environmental cost, and the Tabular ResNet offered a balanced profile. These
results underscore that no single model dominates across all responsibility
criteria, highlighting the necessity of multi-dimensional evaluation for
responsible model selection. Our implementation is available at:
https://github.com/raise-framework/raise.

### Distributed, Parallel, and Cluster Computing

### 1. [SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices](http://arxiv.org/pdf/2510.18544v1)

Authors: Pan Zhou, Yiming Lei, Ling Liu, Xiaoqiong Xu, Ying Cai, Daji Ergu, Hongfang Yu, Yueyue Dai

Large Language Models (LLMs), as the foundational architecture for
next-generation interactive AI applications, not only power intelligent
dialogue systems but also drive the evolution of embodied intelligence on edge
devices, including humanoid robots, smart vehicles, and other scenarios. The
applications running on these edge devices impose differentiated Service Level
Objectives (SLO) requirements on LLM services, specifically manifested as
distinct constraints on Time to First Token (TTFT) and Time Per Output Token
(TPOT) as well as end-to-end latency. Notably, edge devices typically handle
real-time tasks that are extremely sensitive to latency, such as machine
control and navigation planning. However, existing scheduling service systems
still prioritize maximizing output token throughput as the sole optimization
objective, failing to adequately address the diversity of SLO requirements.
This ultimately results in persistently high violation rates for end-to-end
latency or TPOT related SLOs.
  This paper proposes SLICE, an innovative scheduling solution designed for
edge computing scenarios with differentiated SLO requirements. By combining a
utility-maximizing request scheduling algorithm with a dynamic iterative
control mechanism for generation rates, SLICE significantly improves LLM
inference service SLO attainment. Experimental results demonstrate that
compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to
35x higher SLO attainment and 3.4x advantage in task completion time than the
other two solutions.

### 2. [Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](http://arxiv.org/pdf/2510.18586v1)

Authors: Zhuohang Bian, Feiyang Wu, Teng Ma, Youwei Zhuo

Large Language Models (LLMs) are increasingly deployed in complex multi-agent
applications that use external function calls. This workload creates severe
performance challenges for the KV Cache: space contention leads to the eviction
of critical agents' caches and time underutilization leaves the cache of agents
stalled on long-running tool calls idling in GPU memory. We present Tokencake,
a KV-Cache-centric serving framework that co-optimizes scheduling and memory
management with an agent-aware design. Tokencake's Space Scheduler uses dynamic
memory partitioning to shield critical agents from contention, while its Time
Scheduler employs a proactive offload and predictive upload mechanism to
repurpose GPU memory during function call stalls. Our evaluation on
representative multi-agent benchmarks shows that Tokencake can reduce
end-to-end latency by over 47.06%, improve effective GPU memory utilization by
up to 16.9% compared to vLLM.

### 3. [A Distributed Framework for Causal Modeling of Performance Variability in GPU Traces](http://arxiv.org/pdf/2510.18300v1)

Authors: Ankur Lahiry, Ayush Pokharel, Banooqa Banday, Seth Ockerman, Amal Gueroudji, Mohammad Zaeed, Tanzima Z. Islam, Line Pouchard

Large-scale GPU traces play a critical role in identifying performance
bottlenecks within heterogeneous High-Performance Computing (HPC)
architectures. However, the sheer volume and complexity of a single trace of
data make performance analysis both computationally expensive and
time-consuming. To address this challenge, we present an end-to-end parallel
performance analysis framework designed to handle multiple large-scale GPU
traces efficiently. Our proposed framework partitions and processes trace data
concurrently and employs causal graph methods and parallel coordinating chart
to expose performance variability and dependencies across execution flows.
Experimental results demonstrate a 67% improvement in terms of scalability,
highlighting the effectiveness of our pipeline for analyzing multiple traces
independently.

### 4. [Distributed Interactive Proofs for Planarity with Log-Star Communication](http://arxiv.org/pdf/2510.18592v1)

Authors: Yuval Gil, Merav Parter

We provide new communication-efficient distributed interactive proofs for
planarity. The notion of a \emph{distributed interactive proof (DIP)} was
introduced by Kol, Oshman, and Saxena (PODC 2018). In a DIP, the \emph{prover}
is a single centralized entity whose goal is to prove a certain claim regarding
an input graph $G$. To do so, the prover communicates with a distributed
\emph{verifier} that operates concurrently on all $n$ nodes of $G$. A DIP is
measured by the amount of prover-verifier communication it requires. Namely,
the goal is to design a DIP with a small number of interaction rounds and a
small \emph{proof size}, i.e., a small amount of communication per round. Our
main result is an $O(\log ^{*}n)$-round DIP protocol for embedded planarity and
planarity with a proof size of $O(1)$ and $O(\lceil\log \Delta/\log
^{*}n\rceil)$, respectively. In fact, this result can be generalized as
follows. For any $1\leq r\leq \log^{*}n$, there exists an $O(r)$-round protocol
for embedded planarity and planarity with a proof size of $O(\log ^{(r)}n)$ and
$O(\log ^{(r)}n+\log \Delta /r)$, respectively.

### 5. [Towards an Optimized Benchmarking Platform for CI/CD Pipelines](http://arxiv.org/pdf/2510.18640v1)

Authors: Nils Japke, Sebastian Koch, Helmut Lukasczyk, David Bermbach

Performance regressions in large-scale software systems can lead to
substantial resource inefficiencies, making their early detection critical.
Frequent benchmarking is essential for identifying these regressions and
maintaining service-level agreements (SLAs). Performance benchmarks, however,
are resource-intensive and time-consuming, which is a major challenge for
integration into Continuous Integration / Continuous Deployment (CI/CD)
pipelines. Although numerous benchmark optimization techniques have been
proposed to accelerate benchmark execution, there is currently no practical
system that integrates these optimizations seamlessly into real-world CI/CD
pipelines. In this vision paper, we argue that the field of benchmark
optimization remains under-explored in key areas that hinder its broader
adoption. We identify three central challenges to enabling frequent and
efficient benchmarking: (a) the composability of benchmark optimization
strategies, (b) automated evaluation of benchmarking results, and (c) the
usability and complexity of applying these strategies as part of CI/CD systems
in practice. We also introduce a conceptual cloud-based benchmarking framework
handling these challenges transparently. By presenting these open problems, we
aim to stimulate research toward making performance regression detection in
CI/CD systems more practical and effective.

### 6. [MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training](http://arxiv.org/pdf/2510.18830v1)

Authors: Wenxuan Li, Chengruidong Zhang, Huiqiang Jiang, Yucheng Li, Yuqing Yang, Lili Qiu

The adoption of long context windows has become a standard feature in Large
Language Models (LLMs), as extended contexts significantly enhance their
capacity for complex reasoning and broaden their applicability across diverse
scenarios. Dynamic sparse attention is a promising approach for reducing the
computational cost of long-context. However, efficiently training LLMs with
dynamic sparse attention on ultra-long contexts-especially in distributed
settings-remains a significant challenge, due in large part to worker- and
step-level imbalance. This paper introduces MTraining, a novel distributed
methodology leveraging dynamic sparse attention to enable efficient training
for LLMs with ultra-long contexts. Specifically, MTraining integrates three key
components: a dynamic sparse training pattern, balanced sparse ring attention,
and hierarchical sparse ring attention. These components are designed to
synergistically address the computational imbalance and communication overheads
inherent in dynamic sparse attention mechanisms during the training of models
with extensive context lengths. We demonstrate the efficacy of MTraining by
training Qwen2.5-3B, successfully expanding its context window from 32K to 512K
tokens on a cluster of 32 A100 GPUs. Our evaluations on a comprehensive suite
of downstream tasks, including RULER, PG-19, InfiniteBench, and Needle In A
Haystack, reveal that MTraining achieves up to a 6x higher training throughput
while preserving model accuracy. Our code is available at
https://github.com/microsoft/MInference/tree/main/MTraining.

### 7. [PCMS: Parallel Coupler For Multimodel Simulations](http://arxiv.org/pdf/2510.18838v1)

Authors: Jacob S. Merson, Cameron W. Smith, Mark S. Shephard, Fuad Hasan, Abhiyan Paudel, Angel Castillo-Crooke, Joyal Mathew, Mohammad Elahi

This paper presents the Parallel Coupler for Multimodel Simulations (PCMS), a
new GPU accelerated generalized coupling framework for coupling simulation
codes on leadership class supercomputers. PCMS includes distributed control and
field mapping methods for up to five dimensions. For field mapping PCMS can
utilize discretization and field information to accommodate physics
constraints. PCMS is demonstrated with a coupling of the gyrokinetic
microturbulence code XGC with a Monte Carlo neutral transport code DEGAS2 and
with a 5D distribution function coupling of an energetic particle transport
code (GNET) to a gyrokinetic microturbulence code (GTC). Weak scaling is also
demonstrated on up to 2,080 GPUs of Frontier with a weak scaling efficiency of
85%.

### 8. [sNVMe-oF: Secure and Efficient Disaggregated Storage](http://arxiv.org/pdf/2510.18756v1)

Authors: Marcin Chrapek, Meni Orenbach, Ahmad Atamli, Marcin Copik, Fritz Alder, Torsten Hoefler

Disaggregated storage with NVMe-over-Fabrics (NVMe-oF) has emerged as the
standard solution in modern data centers, achieving superior performance,
resource utilization, and power efficiency. Simultaneously, confidential
computing (CC) is becoming the de facto security paradigm, enforcing stronger
isolation and protection for sensitive workloads. However, securing
state-of-the-art storage with traditional CC methods struggles to scale and
compromises performance or security. To address these issues, we introduce
sNVMe-oF, a storage management system extending the NVMe-oF protocol and
adhering to the CC threat model by providing confidentiality, integrity, and
freshness guarantees. sNVMe-oF offers an appropriate control path and novel
concepts such as counter-leasing. sNVMe-oF also optimizes data path performance
by leveraging NVMe metadata, introducing a new disaggregated Hazel Merkle Tree
(HMT), and avoiding redundant IPSec protections. We achieve this without
modifying the NVMe-oF protocol. To prevent excessive resource usage while
delivering line rate, sNVMe-oF also uses accelerators of CC-capable smart NICs.
We prototype sNVMe-oF on an NVIDIA BlueField-3 and demonstrate how it can
achieve as little as 2% performance degradation for synthetic patterns and AI
training.

### 9. [Distributed Allocation and Resource Scheduling Algorithms Resilient to Link Failure](http://arxiv.org/pdf/2510.18273v1)

Authors: Mohammadreza Doostmohammadian, Sergio Pequito

Distributed resource allocation (DRA) is fundamental to modern networked
systems, spanning applications from economic dispatch in smart grids to CPU
scheduling in data centers. Conventional DRA approaches require reliable
communication, yet real-world networks frequently suffer from link failures,
packet drops, and communication delays due to environmental conditions, network
congestion, and security threats.
  We introduce a novel resilient DRA algorithm that addresses these critical
challenges, and our main contributions are as follows: (1) guaranteed
constraint feasibility at all times, ensuring resource-demand balance even
during algorithm termination or network disruption; (2) robust convergence
despite sector-bound nonlinearities at nodes/links, accommodating practical
constraints like quantization and saturation; and (3) optimal performance under
merely uniformly-connected networks, eliminating the need for continuous
connectivity.
  Unlike existing approaches that require persistent network connectivity and
provide only asymptotic feasibility, our graph-theoretic solution leverages
network percolation theory to maintain performance during intermittent
disconnections. This makes it particularly valuable for mobile multi-agent
systems where nodes frequently move out of communication range. Theoretical
analysis and simulations demonstrate that our algorithm converges to optimal
solutions despite heterogeneous time delays and substantial link failures,
significantly advancing the reliability of distributed resource allocation in
practical network environments.

### Digital Libraries

### 1. [ORDENA: ORigin-DEstiNAtion data exploration](http://arxiv.org/pdf/2510.18278v1)

Authors: Karelia Salinas, Victor Barella, André Luiz Cunha, Gabriel Martins de Oliveira, Thales Viera, Luis Gustavo Nonato

Analyzing origin-destination flows is an important problem that has been
extensively investigated in several scientific fields, particularly by the
visualization community. The problem becomes especially challenging when
involving massive data, demanding mechanisms such as data aggregation and
interactive filtering to make the exploratory process doable. However, data
aggregation tends to smooth out certain patterns, and deciding which data
should be filtered is not straightforward. In this work, we propose ORDENA, a
visual analytic tool to explore origin and destination data. ORDENA is built
upon a simple and intuitive scatter plot where the horizontal and vertical axes
correspond to origins and destinations. Therefore, each origin-destination flow
is represented as a point in the scatter plot. How the points are organized in
the plot layout reveals important spatial phenomena present in the data.
Moreover, ORDENA provides explainability resources that allow users to better
understand the relation between origin-destination flows and associated
attributes. We illustrate ORDENA's effectiveness in a set of case studies,
which have also been elaborated in collaboration with domain experts. The
proposed tool has also been evaluated by domain experts not involved in its
development, which provided quite positive feedback about ORDENA.

### Discrete Mathematics

### 1. [Weighted Treedepth is NP-complete on Graphs of Bounded Degree](http://arxiv.org/pdf/2510.18584v1)

Authors: Jona Dirks, Nicole Schirrmacher, Sebastian Siebertz, Alexandre Vigny

A treedepth decomposition of an undirected graph $G$ is a rooted forest $F$
on the vertex set of $G$ such that every edge $uv\in E(G)$ is in
ancestor-descendant relationship in $F$. Given a weight function $w\colon
V(G)\rightarrow \mathbb{N}$, the weighted depth of a treedepth decomposition is
the maximum weight of any path from the root to a leaf, where the weight of a
path is the sum of the weights of its vertices. It is known that deciding
weighted treedepth is NP-complete even on trees. We prove that weighted
treedepth is also NP-complete on bounded degree graphs. On the positive side,
we prove that the problem is efficiently solvable on paths and on 1-subdivided
stars.

### 2. [Brute-force search and Warshall algorithms for matrix-weighted graphs](http://arxiv.org/pdf/2510.18260v1)

Authors: Minh Hoang Trinh, Hyo-Sung Ahn

Although research on the control of networked systems has grown considerably,
graph-theoretic and algorithmic studies on matrix-weighted graphs remain
limited. To bridge this gap in the literature, this work introduces two
algorithms-the brute-force search and the Warshall algorithm-for determining
connectedness and clustering in undirected matrix-weighted graphs. The proposed
algorithms, which are derived from a sufficient condition for connectedness,
emphasize a key distinction between matrix-weighted and scalar-weighted graphs.
While the existence of a path between two vertices guarantees connectedness in
scalar-weighted graphs, connectedness in matrix-weighted graphs is a collective
contribution of all paths joining the two vertices. Proofs of correctness and
numerical examples are provided to illustrate and demonstrate the effectiveness
of the algorithms.

### 3. [Uniformity Testing under User-Level Local Privacy](http://arxiv.org/pdf/2510.18379v1)

Authors: Clément L. Canonne, Abigail Gentle, Vikrant Singhal

We initiate the study of distribution testing under \emph{user-level} local
differential privacy, where each of $n$ users contributes $m$ samples from the
unknown underlying distribution. This setting, albeit very natural, is
significantly more challenging that the usual locally private setting, as for
the same parameter $\varepsilon$ the privacy guarantee must now apply to a full
batch of $m$ data points. While some recent work consider distribution
\emph{learning} in this user-level setting, nothing was known for even the most
fundamental testing task, uniformity testing (and its generalization, identity
testing).
  We address this gap, by providing (nearly) sample-optimal user-level LDP
algorithms for uniformity and identity testing. Motivated by practical
considerations, our main focus is on the private-coin, symmetric setting, which
does not require users to share a common random seed nor to have been assigned
a globally unique identifier.

### 4. [Undirected Multicast Network Coding Gaps via Locally Decodable Codes](http://arxiv.org/pdf/2510.18737v1)

Authors: Mark Braverman, Zhongtian He

The network coding problem asks whether data throughput in a network can be
increased using coding (compared to treating bits as commodities in a flow).
While it is well-known that a network coding advantage exists in directed
graphs, the situation in undirected graphs is much less understood -- in
particular, despite significant effort, it is not even known whether network
coding is helpful at all for unicast sessions.
  In this paper we study the multi-source multicast network coding problem in
undirected graphs. There are $k$ sources broadcasting each to a subset of nodes
in a graph of size $n$. The corresponding combinatorial problem is a version of
the Steiner tree packing problem, and the network coding question asks whether
the multicast coding rate exceeds the tree-packing rate.
  We give the first super-constant bound to this problem, demonstrating an
example with a coding advantage of $\Omega(\log k)$. In terms of graph size, we
obtain a lower bound of $2^{\tilde{\Omega}(\sqrt{\log \log n})}$. We also
obtain an upper bound of $O(\log n)$ on the gap.
  Our main technical contribution is a new reduction that converts
locally-decodable codes in the low-error regime into multicast coding
instances. This gives rise to a new family of explicitly constructed graphs,
which may have other applications.

### Data Structures and Algorithms

### 1. [Nearly Space-Optimal Graph and Hypergraph Sparsification in Insertion-Only Data Streams](http://arxiv.org/pdf/2510.18180v1)

Authors: Vincent Cohen-Addad, David P. Woodruff, Shenghao Xie, Samson Zhou

We study the problem of graph and hypergraph sparsification in insertion-only
data streams. The input is a hypergraph $H=(V, E, w)$ with $n$ nodes, $m$
hyperedges, and rank $r$, and the goal is to compute a hypergraph $\widehat{H}$
that preserves the energy of each vector $x \in \mathbb{R}^n$ in $H$, up to a
small multiplicative error. In this paper, we give a streaming algorithm that
achieves a $(1+\varepsilon)$-approximation, using $\frac{rn}{\varepsilon^2}
\log^2 n \log r \cdot\text{poly}(\log \log m)$ bits of space, matching the
sample complexity of the best known offline algorithm up to $\text{poly}(\log
\log m)$ factors. Our approach also provides a streaming algorithm for graph
sparsification that achieves a $(1+\varepsilon)$-approximation, using
$\frac{n}{\varepsilon^2} \log n \cdot\text{poly}(\log\log n)$ bits of space,
improving the current bound by $\log n$ factors. Furthermore, we give a
space-efficient streaming algorithm for min-cut approximation. Along the way,
we present an online algorithm for $(1+\varepsilon)$-hypergraph sparsification,
which is optimal up to poly-logarithmic factors. As a result, we achieve
$(1+\varepsilon)$-hypergraph sparsification in the sliding window model, with
space optimal up to poly-logarithmic factors. Lastly, we give an adversarially
robust algorithm for hypergraph sparsification using $\frac{n}{\varepsilon^2}
\cdot\text{poly}(r, \log n, \log r, \log \log m)$ bits of space.

### 2. [Static Retrieval Revisited: To Optimality and Beyond](http://arxiv.org/pdf/2510.18237v1)

Authors: Yang Hu, William Kuszmaul, Jingxun Liang, Huacheng Yu, Junkai Zhang, Renfei Zhou

In the static retrieval problem, a data structure must answer retrieval
queries mapping a set of $n$ keys in a universe $[U]$ to $v$-bit values.
Information-theoretically, retrieval data structures can use as little as $nv$
bits of space. For small value sizes $v$, it is possible to achieve $O(1)$
query time while using space $nv + o(n)$ bits -- whether or not such a result
is possible for larger values of $v$ (e.g., $v = \Theta(\log n)$) has remained
open.
  In this paper, we obtain a tight lower bound (as well as matching upper
bounds) for the static retrieval problem. In the case where values are large,
we show that there is actually a significant tension between time and space. It
is not possible, for example, to get $O(1)$ query time using $nv + o(n)$ bits
of space, when $v = \Theta(\log n)$ (and assuming the word RAM model with
$O(\log n)$-bit words).
  At first glance, our lower bound would seem to render retrieval unusable in
many settings that aim to achieve very low redundancy. However, our second
result offers a way around this: We show that, whenever a retrieval data
structure $D_1$ is stored along with another data structure $D_2$ (whose size
is similar to or larger than the size of $D_1$), it is possible to implement
the combined data structure $D_1 \cup D_2$ so that queries to $D_1$ take $O(1)$
time, operations on $D_2$ take the same asymptotic time as if $D_2$ were stored
on its own, and the total space is $nv + \mathrm{Space}(D_2) + n^{0.67}$ bits.

### 3. [Minimum $s$--$t$ Cuts with Fewer Cut Queries](http://arxiv.org/pdf/2510.18274v1)

Authors: Yonggang Jiang, Danupon Nanongkai, Pachara Sawettamalya

We study the problem of computing a minimum $s$--$t$ cut in an unweighted,
undirected graph via \emph{cut queries}. In this model, the input graph is
accessed through an oracle that, given a subset of vertices $S \subseteq V$,
returns the size of the cut $(S, V \setminus S)$.
  This line of work was initiated by Rubinstein, Schramm, and Weinberg (ITCS
2018), who gave a randomized algorithm that computes a minimum $s$--$t$ cut
using $\widetilde{O}(n^{5/3})$ queries, thereby showing that one can avoid
spending $\widetilde{\Theta}(n^2)$ queries required to learn the entire graph.
A recent result by Anand, Saranurak, and Wang (SODA 2025) also matched this
upper bound via a deterministic algorithm based on blocking flows.
  In this work, we present a new randomized algorithm that improves the
cut-query complexity to $\widetilde{O}(n^{8/5})$. At the heart of our approach
is a query-efficient subroutine that incrementally reveals the graph
edge-by-edge while increasing the maximum $s$--$t$ flow in the learned subgraph
at a rate faster than classical augmenting-path methods. Notably, our algorithm
is simple, purely combinatorial, and can be naturally interpreted as a
recursive greedy procedure.
  As a further consequence, we obtain a \emph{deterministic} and
\emph{combinatorial} two-party communication protocol for computing a minimum
$s$--$t$ cut using $\widetilde{O}(n^{11/7})$ bits of communication. This
improves upon the previous best bound of $\widetilde{O}(n^{5/3})$, which was
obtained via reductions from the aforementioned cut-query algorithms. In
parallel, it has been observed that an $\widetilde{O}(n^{3/2})$-bit randomized
protocol can be achieved via continuous optimization techniques; however, these
methods are fundamentally different from our combinatorial approach.

### 4. [Coloring Graphs with Few Colors in the Streaming Model](http://arxiv.org/pdf/2510.18177v1)

Authors: Sepehr Assadi, Janani Sundaresan, Helia Yazdanyar

We study graph coloring problems in the streaming model, where the goal is to
process an $n$-vertex graph whose edges arrive in a stream, using a limited
space that is smaller than the trivial $O(n^2)$ bound. While prior work has
largely focused on coloring graphs with a large number of colors, we explore
the opposite end of the spectrum: deciding whether the input graph can be
colored using only a few, say, a constant number of colors. We are interested
in each of the adversarial, random order, or dynamic streams.
  Our work lays the foundation for this new direction by establishing upper and
lower bounds on space complexity of key variants of the problem. Some of our
main results include:
  - Adversarial: for distinguishing between $q$- vs $2^{\Omega(q)}$-colorable
graphs, lower bounds of $n^{2-o(1)}$ space for $q$ up to
$(\log{n})^{1/2-o(1)}$, and $n^{1+\Omega(1/\log\log{n})}$ space for $q$ further
up to $(\log{n})^{1-o(1)}$.
  - Random order: for distinguishing between $q$- vs $q^t$-colorable graphs for
$q,t \geq 2$, an upper bound of $\tilde{O}(n^{1+1/t})$ space. Specifically,
distinguishing between $q$-colorable graphs vs ones that are not even
poly$(q)$-colorable can be done in $n^{1+o(1)}$ space unlike in adversarial
streams. Although, distinguishing between $q$-colorable vs
$\Omega(q^2)$-colorable graphs requires $\Omega(n^2)$ space even in random
order streams for constant $q$.
  - Dynamic: for distinguishing between $q$- vs $q \cdot t$-colorable graphs
for any $q \geq 3$ and $t \geq 1$, nearly optimal upper and lower bounds of
$\tilde{\Theta}(n^2/t^2)$ space.
  We develop several new technical tools along the way: cluster packing graphs,
a generalization of Ruzsa-Szemer\'edi graphs; a player elimination framework
based on cluster packing graphs; and new edge and vertex sampling lemmas
tailored to graph coloring.

### 5. [Revisiting RFID Missing Tag Identification](http://arxiv.org/pdf/2510.18285v1)

Authors: Kanghuai Liu, Lin Chen, Jihong Yu, Junyi Huang, Shiyuan Liu

We revisit the problem of missing tag identification in RFID networks by
making three contributions. Firstly, we quantitatively compare and gauge the
existing propositions spanning over a decade on missing tag identification. We
show that the expected execution time of the best solution in the literature is
$\Theta \left(N+\frac{(1-\alpha)^2(1-\delta)^2}{ \epsilon^2}\right)$, where
$\delta$ and $\epsilon$ are parameters quantifying the required identification
accuracy, $N$ denotes the number of tags in the system, among which $\alpha N$
tags are missing. Secondly, we analytically establish the expected execution
time lower-bound for any missing tag identification algorithm as
$\Theta\left(\frac{N}{\log N}+\frac{(1-\delta)^2(1-\alpha)^2}{\epsilon^2 \log
\frac{(1-\delta)(1-\alpha)}{\epsilon}}\right)$, thus giving the theoretical
performance limit. Thirdly, we develop a novel missing tag identification
algorithm by leveraging a tree structure with the expected execution time of
$\Theta \left(\frac{\log\log N}{\log N}N+\frac{(1-\alpha)^2(1-\delta)^2}{
\epsilon^2}\right)$, reducing the time overhead by a factor of up to $\log N$
over the best algorithm in the literature. The key technicality in our design
is a novel data structure termed as collision-partition tree (CPT), built on a
subset of bits in tag pseudo-IDs, leading to more balanced tree structure and
reducing the time complexity in parsing the entire tree.

### 6. [Odd and Even Harder Problems on Cycle-Factors](http://arxiv.org/pdf/2510.18393v1)

Authors: Florian Hörsch, Csaba Király, Mirabel Mendoza-Cadena, Gyula Pap, Eszter Szabó, Yutaro Yamaguchi

For a graph (undirected, directed, or mixed), a cycle-factor is a collection
of vertex-disjoint cycles covering the entire vertex set. Cycle-factors subject
to parity constraints arise naturally in the study of structural graph theory
and algorithmic complexity. In this work, we study four variants of the problem
of finding a cycle-factor subject to parity constraints: (1) all cycles are
odd, (2) all cycles are even, (3) at least one cycle is odd, and (4) at least
one cycle is even. These variants are considered in the undirected, directed,
and mixed settings. We show that all but the fourth problem are NP-complete in
all settings, while the complexity of the fourth one remains open for the
directed and undirected cases. We also show that in mixed graphs, even deciding
the existence of any cycle factor is NP-complete.

### 7. [Distributed Interactive Proofs for Planarity with Log-Star Communication](http://arxiv.org/pdf/2510.18592v1)

Authors: Yuval Gil, Merav Parter

We provide new communication-efficient distributed interactive proofs for
planarity. The notion of a \emph{distributed interactive proof (DIP)} was
introduced by Kol, Oshman, and Saxena (PODC 2018). In a DIP, the \emph{prover}
is a single centralized entity whose goal is to prove a certain claim regarding
an input graph $G$. To do so, the prover communicates with a distributed
\emph{verifier} that operates concurrently on all $n$ nodes of $G$. A DIP is
measured by the amount of prover-verifier communication it requires. Namely,
the goal is to design a DIP with a small number of interaction rounds and a
small \emph{proof size}, i.e., a small amount of communication per round. Our
main result is an $O(\log ^{*}n)$-round DIP protocol for embedded planarity and
planarity with a proof size of $O(1)$ and $O(\lceil\log \Delta/\log
^{*}n\rceil)$, respectively. In fact, this result can be generalized as
follows. For any $1\leq r\leq \log^{*}n$, there exists an $O(r)$-round protocol
for embedded planarity and planarity with a proof size of $O(\log ^{(r)}n)$ and
$O(\log ^{(r)}n+\log \Delta /r)$, respectively.

### 8. [An optimal algorithm for average distance in typical regular graphs](http://arxiv.org/pdf/2510.18722v1)

Authors: Alexandros Eskenazis, Manor Mendel, Assaf Naor

We design a deterministic algorithm that, given $n$ points in a
\emph{typical} constant degree regular~graph, queries $O(n)$ distances to
output a constant factor approximation to the average distance among those
points, thus answering a question posed in~\cite{MN14}. Our algorithm uses the
method of~\cite{MN14} to construct a sequence of constant degree graphs that
are expanders with respect to certain nonpositively curved metric spaces,
together with a new rigidity theorem for metric transforms of nonpositively
curved metric spaces. The fact that our algorithm works for typical (uniformly
random) constant degree regular graphs rather than for all constant degree
graphs is unavoidable, thanks to the following impossibility result that we
obtain: For every fixed $k\in \N$, the approximation factor of any algorithm
for average distance that works for all constant degree graphs and queries
$o(n^{1+1/k})$ distances must necessarily be at least $2(k+1)$. This matches
the upper bound attained by the algorithm that was designed for general finite
metric spaces in~\cite{BGS}. Thus, any algorithm for average distance in
constant degree graphs whose approximation guarantee is less than $4$ must
query $\Omega(n^2)$ distances, any such algorithm whose approximation guarantee
is less than $6$ must query $\Omega(n^{3/2})$ distances, any such algorithm
whose approximation guarantee less than $8$ must query $\Omega(n^{4/3})$
distances, and so forth, and furthermore there exist algorithms achieving those
parameters.

### 9. [Uniformity Testing under User-Level Local Privacy](http://arxiv.org/pdf/2510.18379v1)

Authors: Clément L. Canonne, Abigail Gentle, Vikrant Singhal

We initiate the study of distribution testing under \emph{user-level} local
differential privacy, where each of $n$ users contributes $m$ samples from the
unknown underlying distribution. This setting, albeit very natural, is
significantly more challenging that the usual locally private setting, as for
the same parameter $\varepsilon$ the privacy guarantee must now apply to a full
batch of $m$ data points. While some recent work consider distribution
\emph{learning} in this user-level setting, nothing was known for even the most
fundamental testing task, uniformity testing (and its generalization, identity
testing).
  We address this gap, by providing (nearly) sample-optimal user-level LDP
algorithms for uniformity and identity testing. Motivated by practical
considerations, our main focus is on the private-coin, symmetric setting, which
does not require users to share a common random seed nor to have been assigned
a globally unique identifier.

### 10. [LatticeHashForest: An Efficient Data Structure for Repetitive Data and Operations](http://arxiv.org/pdf/2510.18496v1)

Authors: Anamitra Ghorui, Uday P. Khedker

Analysis of entire programs as a single unit, or whole-program analysis,
involves propagation of large amounts of information through the control flow
of the program. This is especially true for pointer analysis, where, unless
significant compromises are made in the precision of the analysis, there is a
combinatorial blowup of information. One of the key problems we observed in our
own efforts is that a lot of duplicate data was being propagated, and many
low-level data structure operations were repeated a large number of times.
  We present what we consider to be a novel and generic data structure,
LatticeHashForest (LHF), to store and operate on such information in a manner
that eliminates a majority of redundant computations and duplicate data in
scenarios similar to those encountered in compilers and program optimization.
LHF differs from similar work in this vein, such as hash-consing, ZDDs, and
BDDs, by not only providing a way to efficiently operate on large, aggregate
structures, but also modifying the elements of such structures in a manner that
they can be deduplicated immediately. LHF also provides a way to perform a
nested construction of elements such that they can be deduplicated at multiple
levels, cutting down the need for additional, nested computations.
  We provide a detailed structural description, along with an abstract model of
this data structure. An entire C++ implementation of LHF is provided as an
artifact along with evaluations of LHF using examples and benchmark programs.
We also supply API documentation and a user manual for users to make
independent applications of LHF. Our main use case in the realm of pointer
analysis shows memory usage reduction to an almost negligible fraction, and
speedups beyond 4x for input sizes approaching 10 million when compared to
other implementations.

### Formal Languages and Automata Theory

### 1. [A Characterization of Turing Machines that Compute Primitive Recursive Functions](http://arxiv.org/pdf/2510.18283v1)

Authors: Daniel G. Schwartz

This paper provides a new and more direct proof of the assertion that a
Turing computable function of the natural numbers is primitive recursive if and
only if the time complexity of the corresponding Turing machine is bounded by a
primitive recursive function of the function's arguments. In addition, it
provides detailed proofs of two consequences of this fact, which, although
well-known in some circles, do not seem to have ever been published. The first
is that the Satisfiability Problem, properly construed as a function of natural
numbers, is primitive recursive. The second is a generalization asserting that
all the problems in NP are similarly primitive recursive. The purpose here is
to present these theorems, fully detailed, in an archival journal, thereby
giving them a status of permanence and general availability.

### 2. [ZipLex: Verified Invertible Lexing with Memoized Derivatives and Zippers](http://arxiv.org/pdf/2510.18479v1)

Authors: Samuel Chassot, Viktor Kunčak

We present ZipLex, a verified framework for invertible lexical analysis.
Unlike past verified lexers that focus only on satisfying the semantics of
regular expressions and the maximal munch property, ZipLex also guarantees that
lexing and printing are mutual inverses. Our design relies on two sets of
ideas: (1) a new abstraction of token sequences that captures the separability
of tokens in a sequence while supporting their efficient manipulation, and (2)
a combination of verified data structures and optimizations, including Huet's
zippers and memoized derivatives, to achieve practical performance. We
implemented ZipLex in Scala and verified its correctness, including
invertibility, using the Stainless verifier. Our evaluation demonstrates that
ZipLex supports realistic applications such as JSON processing and lexers of
programming languages. In comparison to other verified lexers (which do not
enforce invertibility), ZipLex is 4x slower than Coqlex and two orders of
magnitude faster than Verbatim++, showing that verified invertibility can be
achieved without prohibitive cost.

### Graphics

### 1. [MorphModes: Non-rigid Registration via Adaptive Skinning Eigenmodes](http://arxiv.org/pdf/2510.18658v1)

Authors: Gabrielle Browne, Mengfei Liu, Eitan Grinspun, Otman Benchekroun

Non-rigid registration is a crucial task with applications in medical
imaging, industrial robotics, computer vision, and entertainment. Standard
approaches accomplish this task using variations on the Non-Rigid Iterative
Closest Point (NRICP) algorithms, which are prone to local minima and sensitive
to initial conditions. We instead formulate the non-rigid registration problem
as a Signed Distance Function (SDF) matching optimization problem, which
provides richer shape information compared to traditional ICP methods. To avoid
degenerate solutions, we propose to use a smooth Skinning Eigenmode subspace to
parameterize the optimization problem. Finally, we propose an adaptive subspace
optimization scheme to allow the resolution of localized deformations within
the optimization. The result is a non-rigid registration algorithm that is more
robust than NRICP, without the parameter sensitivity present in other
SDF-matching approaches.

### 2. [A Generalizable Light Transport 3D Embedding for Global Illumination](http://arxiv.org/pdf/2510.18189v1)

Authors: Bing Xu, Mukund Varma T, Cheng Wang, Tzumao Li, Lifan Wu, Bartlomiej Wronski, Ravi Ramamoorthi, Marco Salvi

Global illumination (GI) is essential for realistic rendering but remains
computationally expensive due to the complexity of simulating indirect light
transport. Recent neural methods have mainly relied on per-scene optimization,
sometimes extended to handle changes in camera or geometry. Efforts toward
cross-scene generalization have largely stayed in 2D screen space, such as
neural denoising or G-buffer based GI prediction, which often suffer from view
inconsistency and limited spatial understanding. We propose a generalizable 3D
light transport embedding that approximates global illumination directly from
3D scene configurations, without using rasterized or path-traced cues. Each
scene is represented as a point cloud with geometric and material features. A
scalable transformer models global point-to-point interactions to encode these
features into neural primitives. At render time, each query point retrieves
nearby primitives via nearest-neighbor search and aggregates their latent
features through cross-attention to predict the desired rendering quantity. We
demonstrate results on diffuse global illumination prediction across diverse
indoor scenes with varying layouts, geometry, and materials. The embedding
trained for irradiance estimation can be quickly adapted to new rendering tasks
with limited fine-tuning. We also present preliminary results for
spatial-directional radiance field estimation for glossy materials and show how
the normalized field can accelerate unbiased path guiding. This approach
highlights a path toward integrating learned priors into rendering pipelines
without explicit ray-traced illumination cues.

### 3. [ORDENA: ORigin-DEstiNAtion data exploration](http://arxiv.org/pdf/2510.18278v1)

Authors: Karelia Salinas, Victor Barella, André Luiz Cunha, Gabriel Martins de Oliveira, Thales Viera, Luis Gustavo Nonato

Analyzing origin-destination flows is an important problem that has been
extensively investigated in several scientific fields, particularly by the
visualization community. The problem becomes especially challenging when
involving massive data, demanding mechanisms such as data aggregation and
interactive filtering to make the exploratory process doable. However, data
aggregation tends to smooth out certain patterns, and deciding which data
should be filtered is not straightforward. In this work, we propose ORDENA, a
visual analytic tool to explore origin and destination data. ORDENA is built
upon a simple and intuitive scatter plot where the horizontal and vertical axes
correspond to origins and destinations. Therefore, each origin-destination flow
is represented as a point in the scatter plot. How the points are organized in
the plot layout reveals important spatial phenomena present in the data.
Moreover, ORDENA provides explainability resources that allow users to better
understand the relation between origin-destination flows and associated
attributes. We illustrate ORDENA's effectiveness in a set of case studies,
which have also been elaborated in collaboration with domain experts. The
proposed tool has also been evaluated by domain experts not involved in its
development, which provided quite positive feedback about ORDENA.

### 4. [From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation](http://arxiv.org/pdf/2510.18263v1)

Authors: Ziwei Huang, Ying Shu, Hao Fang, Quanyu Long, Wenya Wang, Qiushi Guo, Tiezheng Ge, Leilei Gan

Subject-driven image generation models face a fundamental trade-off between
identity preservation (fidelity) and prompt adherence (editability). While
online reinforcement learning (RL), specifically GPRO, offers a promising
solution, we find that a naive application of GRPO leads to competitive
degradation, as the simple linear aggregation of rewards with static weights
causes conflicting gradient signals and a misalignment with the temporal
dynamics of the diffusion process. To overcome these limitations, we propose
Customized-GRPO, a novel framework featuring two key innovations: (i)
Synergy-Aware Reward Shaping (SARS), a non-linear mechanism that explicitly
penalizes conflicted reward signals and amplifies synergistic ones, providing a
sharper and more decisive gradient. (ii) Time-Aware Dynamic Weighting (TDW),
which aligns the optimization pressure with the model's temporal dynamics by
prioritizing prompt-following in the early, identity preservation in the later.
Extensive experiments demonstrate that our method significantly outperforms
naive GRPO baselines, successfully mitigating competitive degradation. Our
model achieves a superior balance, generating images that both preserve key
identity features and accurately adhere to complex textual prompts.

### Computer Science and Game Theory

### 1. [Contextual Search in Principal-Agent Games: The Curse of Degeneracy](http://arxiv.org/pdf/2510.18567v1)

Authors: Yiding Feng, Mengfan Ma, Bo Peng, Zongqi Wan

In this work, we introduce and study contextual search in general
principal-agent games, where a principal repeatedly interacts with agents by
offering contracts based on contextual information and historical feedback,
without knowing the agents' true costs or rewards. Our model generalizes
classical contextual pricing by accommodating richer agent action spaces. Over
$T$ rounds with $d$-dimensional contexts, we establish an asymptotically tight
exponential $T^{1 - \Theta(1/d)}$ bound in terms of the pessimistic Stackelberg
regret, benchmarked against the best utility for the principal that is
consistent with the observed feedback.
  We also establish a lower bound of $\Omega(T^{\frac{1}{2}-\frac{1}{2d}})$ on
the classic Stackelberg regret for principal-agent games, demonstrating a
surprising double-exponential hardness separation from the contextual pricing
problem (a.k.a, the principal-agent game with two actions), which is known to
admit a near-optimal $O(d\log\log T)$ regret bound [Kleinberg and Leighton,
2003, Leme and Schneider, 2018, Liu et al., 2021]. In particular, this
double-exponential hardness separation occurs even in the special case with
three actions and two-dimensional context. We identify that this significant
increase in learning difficulty arises from a structural phenomenon that we
call contextual action degeneracy, where adversarially chosen contexts can make
some actions strictly dominated (and hence unincentivizable), blocking the
principal's ability to explore or learn about them, and fundamentally limiting
learning progress.

### 2. [Likelihood of the Existence of Average Justified Representation](http://arxiv.org/pdf/2510.18718v1)

Authors: Qishen Han, Biaoshuai Tao, Lirong Xia, Chengkai Zhang, Houyu Zhou

We study the approval-based multi-winner election problem where $n$ voters
jointly decide a committee of $k$ winners from $m$ candidates. We focus on the
axiom \emph{average justified representation} (AJR) proposed by Fernandez,
Elkind, Lackner, Garcia, Arias-Fisteus, Basanta-Val, and Skowron (2017). AJR
postulates that every group of voters with a common preference should be
sufficiently represented in that their average satisfaction should be no less
than their Hare quota. Formally, for every group of
$\lceil\ell\cdot\frac{n}{k}\rceil$ voters with $\ell$ common approved
candidates, the average number of approved winners for this group should be at
least $\ell$. It is well-known that a winning committee satisfying AJR is not
guaranteed to exist for all multi-winner election instances. In this paper, we
study the likelihood of the existence of AJR under the Erd\H{o}s--R\'enyi
model. We consider the Erd\H{o}s--R\'enyi model parameterized by $p\in[0,1]$
that samples multi-winner election instances from the distribution where each
voter approves each candidate with probability $p$ (and the events that voters
approve candidates are independent), and we provide a clean and complete
characterization of the existence of AJR committees in the case where $m$ is a
constant and $n$ tends to infinity. We show that there are two phase transition
points $p_1$ and $p_2$ (with $p_1\leq p_2$) for the parameter $p$ such that: 1)
when $p<p_1$ or $p>p_2$, an AJR committee exists with probability $1-o(1)$, 2)
when $p_1<p<p_2$, an AJR committee exists with probability $o(1)$, and 3) when
$p=p_1$ or $p=p_2$, the probability that an AJR committee exists is bounded
away from both $0$ and $1$.

### 3. [Nash Policy Gradient: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria](http://arxiv.org/pdf/2510.18183v1)

Authors: Eason Yu, Tzu Hao Liu, Yunke Wang, Clément L. Canonne, Nguyen H. Tran, Chang Xu

Finding Nash equilibria in imperfect-information games remains a central
challenge in multi-agent reinforcement learning. While regularization-based
methods have recently achieved last-iteration convergence to a regularized
equilibrium, they require the regularization strength to shrink toward zero to
approximate a Nash equilibrium, often leading to unstable learning in practice.
Instead, we fix the regularization strength at a large value for robustness and
achieve convergence by iteratively refining the reference policy. Our main
theoretical result shows that this procedure guarantees strictly monotonic
improvement and convergence to an exact Nash equilibrium in two-player zero-sum
games, without requiring a uniqueness assumption. Building on this framework,
we develop a practical algorithm, Nash Policy Gradient (NashPG), which
preserves the generalizability of policy gradient methods while relying solely
on the current and reference policies. Empirically, NashPG achieves comparable
or lower exploitability than prior model-free methods on classic benchmark
games and scales to large domains such as Battleship and No-Limit Texas
Hold'em, where NashPG consistently attains higher Elo ratings.

### Human-Computer Interaction

### 1. [Enhancing Urban Data Exploration: Layer Toggling and Visibility-Preserving Lenses for Multi-Attribute Spatial Analysis](http://arxiv.org/pdf/2510.18185v1)

Authors: Karelia Salinas, Luis Gustavo Nonato, Jean-Daniel Fekete, Fernanda Bartolo dos Santos Saran

We propose two novel interaction techniques for visualization-assisted
exploration of urban data: Layer Toggling and Visibility-Preserving Lenses.
Layer Toggling mitigates visual overload by organizing information into
separate layers while enabling comparisons through controlled overlays. This
technique supports focused analysis without losing spatial context and allows
users to switch layers using a dedicated button. Visibility-Preserving Lenses
adapt their size and transparency dynamically, enabling detailed inspection of
dense spatial regions and temporal attributes. These techniques facilitate
urban data exploration and improve prediction. Understanding complex phenomena
related to crime, mobility, and residents' behavior is crucial for informed
urban planning. Yet navigating such data often causes cognitive overload and
visual clutter due to overlapping layers. We validate our visualization tool
through a user study measuring performance, cognitive load, and interaction
efficiency. Using real-world data from Sao Paulo, we demonstrate how our
approach enhances exploratory and analytical tasks and provides guidelines for
future interactive systems.

### 2. [Relief or displacement? How teachers are negotiating generative AI's role in their professional practice](http://arxiv.org/pdf/2510.18296v1)

Authors: Aayushi Dangol, Smriti Kotiyal, Robert Wolfe, Alex J. Bowers, Antonio Vigil, Jason Yip, Julie A. Kientz, Suleman Shahid, Tom Yeh, Vincent Cho, Katie Davis

As generative AI (genAI) rapidly enters classrooms, accompanied by
district-level policy rollouts and industry-led teacher trainings, it is
important to rethink the canonical ``adopt and train'' playbook. Decades of
educational technology research show that tools promising personalization and
access often deepen inequities due to uneven resources, training, and
institutional support. Against this backdrop, we conducted semi-structured
interviews with 22 teachers from a large U.S. school district that was an early
adopter of genAI. Our findings reveal the motivations driving adoption, the
factors underlying resistance, and the boundaries teachers negotiate to align
genAI use with their values. We further contribute by unpacking the
sociotechnical dynamics -- including district policies, professional norms, and
relational commitments -- that shape how teachers navigate the promises and
risks of these tools.

### 3. [Reimagining Disassembly Interfaces with Visualization: Combining Instruction Tracing and Control Flow with DisViz](http://arxiv.org/pdf/2510.18311v1)

Authors: Shadmaan Hye, Matthew P. LeGendre, Katherine E. Isaacs

In applications where efficiency is critical, developers may examine their
compiled binaries, seeking to understand how the compiler transformed their
source code and what performance implications that transformation may have.
This analysis is challenging due to the vast number of disassembled binary
instructions and the many-to-many mappings between them and the source code.
These problems are exacerbated as source code size increases, giving the
compiler more freedom to map and disperse binary instructions across the
disassembly space. Interfaces for disassembly typically display instructions as
an unstructured listing or sacrifice the order of execution. We design a new
visual interface for disassembly code that combines execution order with
control flow structure, enabling analysts to both trace through code and
identify familiar aspects of the computation. Central to our approach is a
novel layout of instructions grouped into basic blocks that displays a looping
structure in an intuitive way. We add to this disassembly representation a
unique block-based mini-map that leverages our layout and shows context across
thousands of disassembly instructions. Finally, we embed our disassembly
visualization in a web-based tool, DisViz, which adds dynamic linking with
source code across the entire application. DizViz was developed in
collaboration with program analysis experts following design study methodology
and was validated through evaluation sessions with ten participants from four
institutions. Participants successfully completed the evaluation tasks,
hypothesized about compiler optimizations, and noted the utility of our new
disassembly view. Our evaluation suggests that our new integrated view helps
application developers in understanding and navigating disassembly code.

### 4. [Khelte Khelte Shikhi: A Proposed HCI Framework for Gamified Interactive Learning with Minecraft in Bangladeshi Education Systems](http://arxiv.org/pdf/2510.18385v1)

Authors: Mohd Ruhul Ameen, Akif Islam, Momen Khandokar Ope

Game-based learning shows real promise for engaging students in well-funded
schools, but what about everyone else? We propose a practical framework for
implementing Minecraft Education Edition in Bangladesh's 130,000 schools where
55 percent lack reliable internet, rural areas experience 12-16 hour daily
power availability, only 8 percent of rural schools have computer access, and
student-teacher ratios reach 52:1. Our approach tackles these constraints
head-on with three deployment tiers: cloud-based multiplayer for urban schools
with stable infrastructure (15 percent), local area network solutions with
solar power for semi-urban contexts (30 percent), and offline turn-based modes
using refurbished hardware for rural settings (55 percent). We provide eight
pre-built curriculum-aligned worlds with complete Bangla localization covering
topics from Lalbagh Fort reconstruction to monsoon flood simulation. The
interface accommodates first-time users through progressive complexity,
culturally familiar metaphors using local farming and architecture, and
accessibility features including keyboard-only controls and 200 percent text
scaling. Teacher training spans 48 hours across digital literacy, pedagogical
integration, and content creation. We detail evaluation protocols with specific
benchmarks: 15 percent learning gains, 70 percent transfer task mastery, System
Usability Scale scores above 70, and sub-two-dollar cost per student-hour. This
framework has not been empirically validated; it synthesizes game-based
learning theory, HCI principles, and contextual analysis to provide
implementable specifications for pilot testing in resource-constrained
settings.

### 5. [Effects of Virtual Controller Representation and Virtuality on Selection Performance in Extended Reality](http://arxiv.org/pdf/2510.18625v1)

Authors: Eric DeDeMarbre, Jay Henderson, J. Felipe Gonzalez, Rob Teather

We present an experiment exploring how the controller's virtual
representation impacts target acquisition performance across MR and VR
contexts. Participants performed selection tasks comparing four visual
configurations: a virtual controller, a virtual hand, both the controller and
the hand, and neither representation. We found performance comparable between
VR and MR, and switching between them did not impact the user's ability to
perform basic tasks. Controller representations mimicking reality enhanced
performance across both modes. However, users perceived performance differently
in MR, indicating the need for unique MR design considerations, particularly
regarding spatial awareness.

### 6. [A Structured Evaluation Framework for Low-Code Platform Selection: A Multi-Criteria Decision Model for Enterprise Digital Transformation](http://arxiv.org/pdf/2510.18590v1)

Authors: Antonio Lamanna

The rapid adoption of Low-Code Development Platforms (LCDPs) has created a
critical need for systematic evaluation methodologies that enable organizations
to make informed platform selection decisions. This paper presents a
comprehensive evaluation framework based on five key criteria: Business Process
Orchestration, UI/UX Customization, Integration and Interoperability,
Governance and Security, and AI-Enhanced Automation. We propose a weighted
scoring model that allows organizations to quantitatively assess and compare
different low-code platforms based on their specific requirements and strategic
priorities. The framework addresses the gap between marketing-driven platform
comparisons and rigorous, context-specific evaluation methodologies. Through
empirical validation in enterprise environments, we demonstrate how this
structured approach can significantly improve decision-making outcomes and
reduce the risk of platform lock-in or inadequate solution selection.

### 7. [KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers](http://arxiv.org/pdf/2510.18355v1)

Authors: Mohd Ruhul Ameen, Akif Islam, Farjana Aktar, M. Saifuzzaman Rafat

In Bangladesh, many farmers continue to face challenges in accessing timely,
expert-level agricultural guidance. This paper presents KrishokBondhu, a
voice-enabled, call-centre-integrated advisory platform built on a
Retrieval-Augmented Generation (RAG) framework, designed specifically for
Bengali-speaking farmers. The system aggregates authoritative agricultural
handbooks, extension manuals, and NGO publications; applies Optical Character
Recognition (OCR) and document-parsing pipelines to digitize and structure the
content; and indexes this corpus in a vector database for efficient semantic
retrieval. Through a simple phone-based interface, farmers can call the system
to receive real-time, context-aware advice: speech-to-text converts the Bengali
query, the RAG module retrieves relevant content, a large language model (Gemma
3-4B) generates a context-grounded response, and text-to-speech delivers the
answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced
high-quality responses for 72.7% of diverse agricultural queries covering crop
management, disease control, and cultivation practices. Compared to the
KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on
a 5-point scale, a 44.7% improvement, with especially large gains in contextual
richness (+367%) and completeness (+100.4%), while maintaining comparable
relevance and technical specificity. Semantic similarity analysis further
revealed a strong correlation between retrieved context and answer quality,
emphasizing the importance of grounding generative responses in curated
documentation. KrishokBondhu demonstrates the feasibility of integrating
call-centre accessibility, multilingual voice interaction, and modern RAG
techniques to deliver expert-level agricultural guidance to remote Bangladeshi
farmers, paving the way toward a fully AI-driven agricultural advisory
ecosystem.

### 8. [One Size Fits All? A Modular Adaptive Sanitization Kit (MASK) for Customizable Privacy-Preserving Phone Scam Detection](http://arxiv.org/pdf/2510.18493v1)

Authors: Kangzhong Wang, Zitong Shen, Youqian Zhang, Michael MK Cheung, Xiapu Luo, Grace Ngai, Eugene Yujun Fu

Phone scams remain a pervasive threat to both personal safety and financial
security worldwide. Recent advances in large language models (LLMs) have
demonstrated strong potential in detecting fraudulent behavior by analyzing
transcribed phone conversations. However, these capabilities introduce notable
privacy risks, as such conversations frequently contain sensitive personal
information that may be exposed to third-party service providers during
processing. In this work, we explore how to harness LLMs for phone scam
detection while preserving user privacy. We propose MASK (Modular Adaptive
Sanitization Kit), a trainable and extensible framework that enables dynamic
privacy adjustment based on individual preferences. MASK provides a pluggable
architecture that accommodates diverse sanitization methods - from traditional
keyword-based techniques for high-privacy users to sophisticated neural
approaches for those prioritizing accuracy. We also discuss potential modeling
approaches and loss function designs for future development, enabling the
creation of truly personalized, privacy-aware LLM-based detection systems that
balance user trust and detection effectiveness, even beyond phone scam context.

### Information Retrieval

### 1. [Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights](http://arxiv.org/pdf/2510.18277v1)

Authors: Nikolaos Belibasakis, Anastasios Giannaros, Ioanna Giannoukou, Spyros Sioutas

The increasing number of data a booking platform such as Booking.com and
AirBnB offers make it challenging for interested parties to browse through the
available accommodations and analyze reviews in an efficient way. Efforts have
been made from the booking platform providers to utilize recommender systems in
an effort to enable the user to filter the results by factors such as stars,
amenities, cost but most valuable insights can be provided by the unstructured
text-based reviews. Going through these reviews one-by-one requires a
substantial amount of time to be devoted while a respectable percentage of the
reviews won't provide to the user what they are actually looking for.
  This research publication explores how Large Language Models (LLMs) can
enhance short rental apartments recommendations by summarizing and mining key
insights from user reviews. The web application presented in this paper, named
"instaGuide", automates the procedure of isolating the text-based user reviews
from a property on the Booking.com platform, synthesizing the summary of the
reviews, and enabling the user to query specific aspects of the property in an
effort to gain feedback on their personal questions/criteria.
  During the development of the instaGuide tool, numerous LLM models were
evaluated based on accuracy, cost, and response quality. The results suggest
that the LLM-powered summarization reduces significantly the amount of time the
users need to devote on their search for the right short rental apartment,
improving the overall decision-making procedure.

### 2. [LLMs as Sparse Retrievers:A Framework for First-Stage Product Search](http://arxiv.org/pdf/2510.18527v1)

Authors: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Sen Li, Wenjun Peng, Fuyu Lv, Xueqi Cheng

Product search is a crucial component of modern e-commerce platforms, with
billions of user queries every day. In product search systems, first-stage
retrieval should achieve high recall while ensuring efficient online
deployment. Sparse retrieval is particularly attractive in this context due to
its interpretability and storage efficiency. However, sparse retrieval methods
suffer from severe vocabulary mismatch issues, leading to suboptimal
performance in product search scenarios.With their potential for semantic
analysis, large language models (LLMs) offer a promising avenue for mitigating
vocabulary mismatch issues and thereby improving retrieval quality. Directly
applying LLMs to sparse retrieval in product search exposes two key
challenges:(1)Queries and product titles are typically short and highly
susceptible to LLM-induced hallucinations, such as generating irrelevant
expansion terms or underweighting critical literal terms like brand names and
model numbers;(2)The large vocabulary space of LLMs leads to difficulty in
initializing training effectively, making it challenging to learn meaningful
sparse representations in such ultra-high-dimensional spaces.To address these
challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs
as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that
alleviates hallucination in lexical expansion by reinforcing underweighted
literal terms through a residual compensation mechanism; and (2)A lexical
focusing window that facilitates effective training initialization via a
coarse-to-fine sparsification strategy.Extensive offline and online experiments
show that PROSPER significantly outperforms sparse baselines and achieves
recall performance comparable to advanced dense retrievers, while also
achieving revenue increments online.

### 3. [LIME: Link-based user-item Interaction Modeling with decoupled xor attention for Efficient test time scaling](http://arxiv.org/pdf/2510.18239v1)

Authors: Yunjiang Jiang, Ayush Agarwal, Yang Liu, Bi Xue

Scaling large recommendation systems requires advancing three major
frontiers: processing longer user histories, expanding candidate sets, and
increasing model capacity. While promising, transformers' computational cost
scales quadratically with the user sequence length and linearly with the number
of candidates. This trade-off makes it prohibitively expensive to expand
candidate sets or increase sequence length at inference, despite the
significant performance improvements.
  We introduce \textbf{LIME}, a novel architecture that resolves this
trade-off. Through two key innovations, LIME fundamentally reduces
computational complexity. First, low-rank ``link embeddings" enable
pre-computation of attention weights by decoupling user and candidate
interactions, making the inference cost nearly independent of candidate set
size. Second, a linear attention mechanism, \textbf{LIME-XOR}, reduces the
complexity with respect to user sequence length from quadratic ($O(N^2)$) to
linear ($O(N)$).
  Experiments on public and industrial datasets show LIME achieves near-parity
with state-of-the-art transformers but with a 10$\times$ inference speedup on
large candidate sets or long sequence lengths. When tested on a major
recommendation platform, LIME improved user engagement while maintaining
minimal inference costs with respect to candidate set size and user history
length, establishing a new paradigm for efficient and expressive recommendation
systems.

### 4. [Evaluating LLM-Based Mobile App Recommendations: An Empirical Study](http://arxiv.org/pdf/2510.18364v1)

Authors: Quim Motger, Xavier Franch, Vincenzo Gervasi, Jordi Marco

Large Language Models (LLMs) are increasingly used to recommend mobile
applications through natural language prompts, offering a flexible alternative
to keyword-based app store search. Yet, the reasoning behind these
recommendations remains opaque, raising questions about their consistency,
explainability, and alignment with traditional App Store Optimization (ASO)
metrics. In this paper, we present an empirical analysis of how widely-used
general purpose LLMs generate, justify, and rank mobile app recommendations.
Our contributions are: (i) a taxonomy of 16 generalizable ranking criteria
elicited from LLM outputs; (ii) a systematic evaluation framework to analyse
recommendation consistency and responsiveness to explicit ranking instructions;
and (iii) a replication package to support reproducibility and future research
on AI-based recommendation systems. Our findings reveal that LLMs rely on a
broad yet fragmented set of ranking criteria, only partially aligned with
standard ASO metrics. While top-ranked apps tend to be consistent across runs,
variability increases with ranking depth and search specificity. LLMs exhibit
varying sensitivity to explicit ranking instructions - ranging from substantial
adaptations to near-identical outputs - highlighting their complex reasoning
dynamics in conversational app discovery. Our results aim to support end-users,
app developers, and recommender-systems researchers in navigating the emerging
landscape of conversational app discovery.

### 5. [KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers](http://arxiv.org/pdf/2510.18355v1)

Authors: Mohd Ruhul Ameen, Akif Islam, Farjana Aktar, M. Saifuzzaman Rafat

In Bangladesh, many farmers continue to face challenges in accessing timely,
expert-level agricultural guidance. This paper presents KrishokBondhu, a
voice-enabled, call-centre-integrated advisory platform built on a
Retrieval-Augmented Generation (RAG) framework, designed specifically for
Bengali-speaking farmers. The system aggregates authoritative agricultural
handbooks, extension manuals, and NGO publications; applies Optical Character
Recognition (OCR) and document-parsing pipelines to digitize and structure the
content; and indexes this corpus in a vector database for efficient semantic
retrieval. Through a simple phone-based interface, farmers can call the system
to receive real-time, context-aware advice: speech-to-text converts the Bengali
query, the RAG module retrieves relevant content, a large language model (Gemma
3-4B) generates a context-grounded response, and text-to-speech delivers the
answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced
high-quality responses for 72.7% of diverse agricultural queries covering crop
management, disease control, and cultivation practices. Compared to the
KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on
a 5-point scale, a 44.7% improvement, with especially large gains in contextual
richness (+367%) and completeness (+100.4%), while maintaining comparable
relevance and technical specificity. Semantic similarity analysis further
revealed a strong correlation between retrieved context and answer quality,
emphasizing the importance of grounding generative responses in curated
documentation. KrishokBondhu demonstrates the feasibility of integrating
call-centre accessibility, multilingual voice interaction, and modern RAG
techniques to deliver expert-level agricultural guidance to remote Bangladeshi
farmers, paving the way toward a fully AI-driven agricultural advisory
ecosystem.

### 6. [Censorship Chokepoints: New Battlegrounds for Regional Surveillance, Censorship and Influence on the Internet](http://arxiv.org/pdf/2510.18394v1)

Authors: Yong Zhang, Nishanth Sastry

Undoubtedly, the Internet has become one of the most important conduits to
information for the general public. Nonetheless, Internet access can be and has
been limited systematically or blocked completely during political events in
numerous countries and regions by various censorship mechanisms. Depending on
where the core filtering component is situated, censorship techniques have been
classified as client-based, server-based, or network-based. However, as the
Internet evolves rapidly, new and sophisticated censorship techniques have
emerged, which involve techniques that cut across locations and involve new
forms of hurdles to information access. We argue that modern censorship can be
better understood through a new lens that we term chokepoints, which identifies
bottlenecks in the content production or delivery cycle where efficient new
forms of large-scale client-side surveillance and filtering mechanisms have
emerged.

### 7. [ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization](http://arxiv.org/pdf/2510.18433v1)

Authors: Yuanhe Guo, Linxi Xie, Zhuoran Chen, Kangrui Yu, Ryan Po, Guandao Yang, Gordon Wetztein, Hongyi Wen

We introduce ImageGem, a dataset for studying generative models that
understand fine-grained individual preferences. We posit that a key challenge
hindering the development of such a generative model is the lack of in-the-wild
and fine-grained user preference annotations. Our dataset features real-world
interaction data from 57K users, who collectively have built 242K customized
LoRAs, written 3M text prompts, and created 5M generated images. With user
preference annotations from our dataset, we were able to train better
preference alignment models. In addition, leveraging individual user
preference, we investigated the performance of retrieval models and a
vision-language model on personalized image retrieval and generative model
recommendation. Finally, we propose an end-to-end framework for editing
customized diffusion models in a latent weight space to align with individual
user preferences. Our results demonstrate that the ImageGem dataset enables,
for the first time, a new paradigm for generative model personalization.

### Machine Learning

### 1. [Joint Optimization of Cooperation Efficiency and Communication Covertness for Target Detection with AUVs](http://arxiv.org/pdf/2510.18225v1)

Authors: Xueyao Zhang, Bo Yang, Zhiwen Yu, Xuelin Cao, Wei Xiang, Bin Guo, Liang Wang, Billy Pik Lik Lau, George C. Alexandropoulos, Jun Luo, Mérouane Debbah, Zhu Han, Chau Yuen

This paper investigates underwater cooperative target detection using
autonomous underwater vehicles (AUVs), with a focus on the critical trade-off
between cooperation efficiency and communication covertness. To tackle this
challenge, we first formulate a joint trajectory and power control optimization
problem, and then present an innovative hierarchical action management
framework to solve it. According to the hierarchical formulation, at the macro
level, the master AUV models the agent selection process as a Markov decision
process and deploys the proximal policy optimization algorithm for strategic
task allocation. At the micro level, each selected agent's decentralized
decision-making is modeled as a partially observable Markov decision process,
and a multi-agent proximal policy optimization algorithm is used to dynamically
adjust its trajectory and transmission power based on its local observations.
Under the centralized training and decentralized execution paradigm, our target
detection framework enables adaptive covert cooperation while satisfying both
energy and mobility constraints. By comprehensively modeling the considered
system, the involved signals and tasks, as well as energy consumption,
theoretical insights and practical solutions for the efficient and secure
operation of multiple AUVs are provided, offering significant implications for
the execution of underwater covert communication tasks.

### 2. [Towards Fast LLM Fine-tuning through Zeroth-Order Optimization with Projected Gradient-Aligned Perturbations](http://arxiv.org/pdf/2510.18228v1)

Authors: Zhendong Mi, Qitao Tan, Grace Li Zhang, Zhaozhuo Xu, Geng Yuan, Shaoyi Huang

Fine-tuning large language models (LLMs) using zeroth-order (ZO) optimization
has emerged as a promising alternative to traditional gradient-based methods
due to its reduced memory footprint requirement. However, existing ZO methods
suffer from high variance in gradient estimation, leading to slow convergence
and suboptimal performance on large-scale models. In this work, we propose
P-GAP, a fast LLM fine-tuning approach through zeroth-order optimization with
Projected Gradient-Aligned Perturbations. Specifically, we first estimate a
low-dimensional gradient space and then align perturbations in projected
gradients' direction within the space. This approach enables reduced the number
of perturbed parameters and decreased variance, therefore accelerated
convergence for LLM fine-tuning. Experiments on LLMs show that P-GAP
consistently surpasses the baselines, achieving up to 6% increase in accuracy
on classification tasks and up to 12% higher accuracy on generation tasks, with
up to about 81% less training iterations and 70% less GPU hours. These results
demonstrate that P-GAP enables fast, scalable, and resource-efficient ZO LLM
fine-tuning.

### 3. [Learning with Dual-level Noisy Correspondence for Multi-modal Entity Alignment](http://arxiv.org/pdf/2510.18240v1)

Authors: Haobin Li, Yijie Lin, Peng Hu, Mouxing Yang, Xi Peng

Multi-modal entity alignment (MMEA) aims to identify equivalent entities
across heterogeneous multi-modal knowledge graphs (MMKGs), where each entity is
described by attributes from various modalities. Existing methods typically
assume that both intra-entity and inter-graph correspondences are faultless,
which is often violated in real-world MMKGs due to the reliance on expert
annotations. In this paper, we reveal and study a highly practical yet
under-explored problem in MMEA, termed Dual-level Noisy Correspondence (DNC).
DNC refers to misalignments in both intra-entity (entity-attribute) and
inter-graph (entity-entity and attribute-attribute) correspondences. To address
the DNC problem, we propose a robust MMEA framework termed RULE. RULE first
estimates the reliability of both intra-entity and inter-graph correspondences
via a dedicated two-fold principle. Leveraging the estimated reliabilities,
RULE mitigates the negative impact of intra-entity noise during attribute
fusion and prevents overfitting to noisy inter-graph correspondences during
inter-graph discrepancy elimination. Beyond the training-time designs, RULE
further incorporates a correspondence reasoning module that uncovers the
underlying attribute-attribute connection across graphs, guaranteeing more
accurate equivalent entity identification. Extensive experiments on five
benchmarks verify the effectiveness of our method against the DNC compared with
seven state-of-the-art methods.The code is available at
\href{https://github.com/XLearning-SCU/RULE}{XLearning-SCU/RULE}

### 4. [Online Time Series Forecasting with Theoretical Guarantees](http://arxiv.org/pdf/2510.18281v1)

Authors: Zijian Li, Changze Zhou, Minghao Fu, Sanjay Manjunath, Fan Feng, Guangyi Chen, Yingyao Hu, Ruichu Cai, Kun Zhang

This paper is concerned with online time series forecasting, where unknown
distribution shifts occur over time, i.e., latent variables influence the
mapping from historical to future observations. To develop an automated way of
online time series forecasting, we propose a Theoretical framework for Online
Time-series forecasting (TOT in short) with theoretical guarantees.
Specifically, we prove that supplying a forecaster with latent variables
tightens the Bayes risk, the benefit endures under estimation uncertainty of
latent variables and grows as the latent variables achieve a more precise
identifiability. To better introduce latent variables into online forecasting
algorithms, we further propose to identify latent variables with minimal
adjacent observations. Based on these results, we devise a model-agnostic
blueprint by employing a temporal decoder to match the distribution of observed
variables and two independent noise estimators to model the causal inference of
latent variables and mixing procedures of observed variables, respectively.
Experiment results on synthetic data support our theoretical claims. Moreover,
plug-in implementations built on several baselines yield general improvement
across multiple benchmarks, highlighting the effectiveness in real-world
applications.

### 5. [Physics-Informed Parametric Bandits for Beam Alignment in mmWave Communications](http://arxiv.org/pdf/2510.18299v1)

Authors: Hao Qin, Thang Duong, Ming Li, Chicheng Zhang

In millimeter wave (mmWave) communications, beam alignment and tracking are
crucial to combat the significant path loss. As scanning the entire directional
space is inefficient, designing an efficient and robust method to identify the
optimal beam directions is essential. Since traditional bandit algorithms
require a long time horizon to converge under large beam spaces, many existing
works propose efficient bandit algorithms for beam alignment by relying on
unimodality or multimodality assumptions on the reward function's structure.
However, such assumptions often do not hold (or cannot be strictly satisfied)
in practice, which causes such algorithms to converge to choosing suboptimal
beams.
  In this work, we propose two physics-informed bandit algorithms
\textit{pretc} and \textit{prgreedy} that exploit the sparse multipath property
of mmWave channels - a generic but realistic assumption - which is connected to
the Phase Retrieval Bandit problem. Our algorithms treat the parameters of each
path as black boxes and maintain optimal estimates of them based on sampled
historical rewards. \textit{pretc} starts with a random exploration phase and
then commits to the optimal beam under the estimated reward function.
\textit{prgreedy} performs such estimation in an online manner and chooses the
best beam under current estimates. Our algorithms can also be easily adapted to
beam tracking in the mobile setting. Through experiments using both the
synthetic DeepMIMO dataset and the real-world DeepSense6G dataset, we
demonstrate that both algorithms outperform existing approaches in a wide range
of scenarios across diverse channel environments, showing their
generalizability and robustness.

### 6. [Why Policy Gradient Algorithms Work for Undiscounted Total-Reward MDPs](http://arxiv.org/pdf/2510.18340v1)

Authors: Jongmin Lee, Ernest K. Ryu

The classical policy gradient method is the theoretical and conceptual
foundation of modern policy-based reinforcement learning (RL) algorithms. Most
rigorous analyses of such methods, particularly those establishing convergence
guarantees, assume a discount factor $\gamma < 1$. In contrast, however, a
recent line of work on policy-based RL for large language models uses the
undiscounted total-reward setting with $\gamma = 1$, rendering much of the
existing theory inapplicable. In this paper, we provide analyses of the policy
gradient method for undiscounted expected total-reward infinite-horizon MDPs
based on two key insights: (i) the classification of the MDP states into
recurrent and transient states is invariant over the set of policies that
assign strictly positive probability to every action (as is typical in deep RL
models employing a softmax output layer) and (ii) the classical state
visitation measure (which may be ill-defined when $\gamma = 1$) can be replaced
with a new object that we call the transient visitation measure.

### 7. [Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](http://arxiv.org/pdf/2510.18360v1)

Authors: Sunwoo Kim, Hyunjin Hwang, Kijung Shin

The performance of a deep learning model on a specific task and dataset
depends heavily on its neural architecture, motivating considerable efforts to
rapidly and accurately identify architectures suited to the target task and
dataset. To achieve this, researchers use machine learning models-typically
neural architecture encoders-to predict the performance of a neural
architecture. Many state-of-the-art encoders aim to capture information flow
within a neural architecture, which reflects how information moves through the
forward pass and backpropagation, via a specialized model structure. However,
due to their complicated structures, these flow-based encoders are
significantly slower to process neural architectures compared to simpler
encoders, presenting a notable practical challenge. To address this, we propose
FGP, a novel pre-training method for neural architecture encoding that trains
an encoder to capture the information flow without requiring specialized model
structures. FGP trains an encoder to reconstruct a flow surrogate, our proposed
representation of the neural architecture's information flow. Our experiments
show that FGP boosts encoder performance by up to 106% in Precision-1%,
compared to the same encoder trained solely with supervised learning.

### 8. [Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](http://arxiv.org/pdf/2510.18363v1)

Authors: Zhen Zhang, Bingsheng He

Unsupervised Graph Domain Adaptation has become a promising paradigm for
transferring knowledge from a fully labeled source graph to an unlabeled target
graph. Existing graph domain adaptation models primarily focus on the
closed-set setting, where the source and target domains share the same label
spaces. However, this assumption might not be practical in the real-world
scenarios, as the target domain might include classes that are not present in
the source domain. In this paper, we investigate the problem of unsupervised
open-set graph domain adaptation, where the goal is to not only correctly
classify target nodes into the known classes, but also recognize previously
unseen node types into the unknown class. Towards this end, we propose a novel
framework called GraphRTA, which conducts reprogramming on both the graph and
model sides. Specifically, we reprogram the graph by modifying target graph
structure and node features, which facilitates better separation of known and
unknown classes. Meanwhile, we also perform model reprogramming by pruning
domain-specific parameters to reduce bias towards the source graph while
preserving parameters that capture transferable patterns across graphs.
Additionally, we extend the classifier with an extra dimension for the unknown
class, thus eliminating the need of manually specified threshold in open-set
recognition. Comprehensive experiments on several public datasets demonstrate
that our proposed model can achieve satisfied performance compared with recent
state-of-the-art baselines. Our source codes and datasets are publicly
available at https://github.com/cszhangzhen/GraphRTA.

### 9. [Training Diverse Graph Experts for Ensembles: A Systematic Empirical Study](http://arxiv.org/pdf/2510.18370v1)

Authors: Gangda Deng, Yuxin Yang, Ömer Faruk Akgül, Hanqing Zeng, Yinglong Xia, Rajgopal Kannan, Viktor Prasanna

Graph Neural Networks (GNNs) have become essential tools for learning on
relational data, yet the performance of a single GNN is often limited by the
heterogeneity present in real-world graphs. Recent advances in
Mixture-of-Experts (MoE) frameworks demonstrate that assembling multiple,
explicitly diverse GNNs with distinct generalization patterns can significantly
improve performance. In this work, we present the first systematic empirical
study of expert-level diversification techniques for GNN ensembles. Evaluating
20 diversification strategies -- including random re-initialization,
hyperparameter tuning, architectural variation, directionality modeling, and
training data partitioning -- across 14 node classification benchmarks, we
construct and analyze over 200 ensemble variants. Our comprehensive evaluation
examines each technique in terms of expert diversity, complementarity, and
ensemble performance. We also uncovers mechanistic insights into training
maximally diverse experts. These findings provide actionable guidance for
expert training and the design of effective MoE frameworks on graph data. Our
code is available at https://github.com/Hydrapse/bench-gnn-diversification.

### 10. [Approximation Rates of Shallow Neural Networks: Barron Spaces, Activation Functions and Optimality Analysis](http://arxiv.org/pdf/2510.18388v1)

Authors: Jian Lu, Xiaohuang Huang

This paper investigates the approximation properties of shallow neural
networks with activation functions that are powers of exponential functions. It
focuses on the dependence of the approximation rate on the dimension and the
smoothness of the function being approximated within the Barron function space.
We examine the approximation rates of ReLU$^{k}$ activation functions, proving
that the optimal rate cannot be achieved under $\ell^{1}$-bounded coefficients
or insufficient smoothness conditions.
  We also establish optimal approximation rates in various norms for functions
in Barron spaces and Sobolev spaces, confirming the curse of dimensionality.
Our results clarify the limits of shallow neural networks' approximation
capabilities and offer insights into the selection of activation functions and
network structures.

### Neural and Evolutionary Computing

### 1. [The Emergence of Complex Behavior in Large-Scale Ecological Environments](http://arxiv.org/pdf/2510.18221v1)

Authors: Joseph Bejjani, Chase Van Amburg, Chengrui Wang, Chloe Huangyuan Su, Sarah M. Pratt, Yasin Mazloumi, Naeem Khoshnevis, Sham M. Kakade, Kianté Brantley

We explore how physical scale and population size shape the emergence of
complex behaviors in open-ended ecological environments. In our setting, agents
are unsupervised and have no explicit rewards or learning objectives but
instead evolve over time according to reproduction, mutation, and natural
selection. As they act, agents also shape their environment and the population
around them in an ongoing dynamic ecology. Our goal is not to optimize a single
high-performance policy, but instead to examine how behaviors emerge and evolve
across large populations due to natural competition and environmental
pressures. In an effort to discover how complex behaviors naturally emerge, we
conduct experiments in large-scale worlds that reach populations of more than
60,000 individual agents, each with their own evolved neural network policy. We
identify various emergent behaviors such as long-range resource extraction,
vision-based foraging, and predation that arise under competitive and survival
pressures. We examine how sensing modalities and environmental scale affect the
emergence of these behaviors, finding that some appear only in sufficiently
large environments and populations, with larger scales increasing behavioral
stability and consistency. While there is a rich history of research in
evolutionary settings, our scaling results provide promising new directions to
explore ecology as an instrument of machine learning in an era of abundant
computational resources. Experimental code is available at
https://github.com/jbejjani2022/ecological-emergent-behavior.

### 2. [Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](http://arxiv.org/pdf/2510.18650v1)

Authors: Kyo Kuroki, Yasuyuki Okoshi, Thiem Van Chu, Kazushi Kawamura, Masato Motomura

This paper proposes a novel matrix quantization method, Binary Quadratic
Quantization (BQQ). In contrast to conventional first-order quantization
approaches, such as uniform quantization and binary coding quantization, that
approximate real-valued matrices via linear combinations of binary bases, BQQ
leverages the expressive power of binary quadratic expressions while
maintaining an extremely compact data format. We validate our approach with two
experiments: a matrix compression benchmark and post-training quantization
(PTQ) on pretrained Vision Transformer-based models. Experimental results
demonstrate that BQQ consistently achieves a superior trade-off between memory
efficiency and reconstruction error than conventional methods for compressing
diverse matrix data. It also delivers strong PTQ performance, even though we
neither target state-of-the-art PTQ accuracy under tight memory constraints nor
rely on PTQ-specific binary matrix optimization. For example, our proposed
method outperforms the state-of-the-art PTQ method by up to 2.2\% and 59.1% on
the ImageNet dataset under the calibration-based and data-free scenarios,
respectively, with quantization equivalent to 2 bits. These findings highlight
the surprising effectiveness of binary quadratic expressions for efficient
matrix approximation and neural network compression.

### 3. [Symbolic Emulators for Cosmology: Accelerating Cosmological Analyses Without Sacrificing Precision](http://arxiv.org/pdf/2510.18749v1)

Authors: Deaglan J. Bartlett, Shivam Pandey

In cosmology, emulators play a crucial role by providing fast and accurate
predictions of complex physical models, enabling efficient exploration of
high-dimensional parameter spaces that would be computationally prohibitive
with direct numerical simulations. Symbolic emulators have emerged as promising
alternatives to numerical approaches, delivering comparable accuracy with
significantly faster evaluation times. While previous symbolic emulators were
limited to relatively narrow prior ranges, we expand these to cover the
parameter space relevant for current cosmological analyses. We introduce
approximations to hypergeometric functions used for the $\Lambda$CDM comoving
distance and linear growth factor which are accurate to better than 0.001% and
0.05%, respectively, for all redshifts and for $\Omega_{\rm m} \in [0.1, 0.5]$.
We show that integrating symbolic emulators into a Dark Energy Survey-like
$3\times2$pt analysis produces cosmological constraints consistent with those
obtained using standard numerical methods. Our symbolic emulators offer
substantial improvements in speed and memory usage, demonstrating their
practical potential for scalable, likelihood-based inference.

### Networking and Internet Architecture

### 1. [JAUNT: Joint Alignment of User Intent and Network State for QoE-centric LLM Tool Routing](http://arxiv.org/pdf/2510.18550v1)

Authors: Enhan Li, Hongyang Du

Large Language Models (LLMs) increasingly rely on emerging protocols such as
the Model Context Protocol (MCP) to invoke external tools and services.
However, current tool routing mechanisms remain fragile because they only
consider functional matching between users' queries and tools. In practice,
user intent expressed through queries can be vague or underspecified, and the
actual Quality of Experience (QoE) also depends on external factors such as
link latency and server availability that are not captured by semantics alone.
To address this challenge, we propose JAUNT, a framework for Joint Alignment of
User intent and Network state in QoE-centric Tool routing. JAUNT introduces a
dual-view alignment strategy that interprets user intent while employing LLM
agents to construct network profiles, mapping numerical performance indicators
into the semantic space to guide routing. We further design a benchmark that
integrates diverse user request patterns with heterogeneous network states,
enabling systematic evaluation of QoE outcomes. Experimental results show that
JAUNT significantly improves QoE compared with several baselines, demonstrating
the importance of aligning both intent and network state for scalable LLM
service orchestration.

### 2. [Revisiting RFID Missing Tag Identification](http://arxiv.org/pdf/2510.18285v1)

Authors: Kanghuai Liu, Lin Chen, Jihong Yu, Junyi Huang, Shiyuan Liu

We revisit the problem of missing tag identification in RFID networks by
making three contributions. Firstly, we quantitatively compare and gauge the
existing propositions spanning over a decade on missing tag identification. We
show that the expected execution time of the best solution in the literature is
$\Theta \left(N+\frac{(1-\alpha)^2(1-\delta)^2}{ \epsilon^2}\right)$, where
$\delta$ and $\epsilon$ are parameters quantifying the required identification
accuracy, $N$ denotes the number of tags in the system, among which $\alpha N$
tags are missing. Secondly, we analytically establish the expected execution
time lower-bound for any missing tag identification algorithm as
$\Theta\left(\frac{N}{\log N}+\frac{(1-\delta)^2(1-\alpha)^2}{\epsilon^2 \log
\frac{(1-\delta)(1-\alpha)}{\epsilon}}\right)$, thus giving the theoretical
performance limit. Thirdly, we develop a novel missing tag identification
algorithm by leveraging a tree structure with the expected execution time of
$\Theta \left(\frac{\log\log N}{\log N}N+\frac{(1-\alpha)^2(1-\delta)^2}{
\epsilon^2}\right)$, reducing the time overhead by a factor of up to $\log N$
over the best algorithm in the literature. The key technicality in our design
is a novel data structure termed as collision-partition tree (CPT), built on a
subset of bits in tag pseudo-IDs, leading to more balanced tree structure and
reducing the time complexity in parsing the entire tree.

### 3. [How2Compress: Scalable and Efficient Edge Video Analytics via Adaptive Granular Video Compression](http://arxiv.org/pdf/2510.18409v1)

Authors: Yuheng Wu, Thanh-Tung Nguyen, Lucas Liebe, Quang Tau, Pablo Espinosa Campos, Jinghan Cheng, Dongman Lee

With the rapid proliferation of the Internet of Things, video analytics has
become a cornerstone application in wireless multimedia sensor networks. To
support such applications under bandwidth constraints, learning-based adaptive
quantization for video compression have demonstrated strong potential in
reducing bitrate while maintaining analytical accuracy. However, existing
frameworks often fail to fully exploit the fine-grained quality control enabled
by modern blockbased video codecs, leaving significant compression efficiency
untapped.
  In this paper, we present How2Compress, a simple yet effective framework
designed to enhance video compression efficiency through precise, fine-grained
quality control at the macroblock level. How2Compress is a plug-and-play module
and can be seamlessly integrated into any existing edge video analytics
pipelines. We implement How2Compress on the H.264 codec and evaluate its
performance across diverse real-world scenarios. Experimental results show that
How2Compress achieves up to $50.4\%$ bitrate savings and outperforms baselines
by up to $3.01\times$ without compromising accuracy, demonstrating its
practical effectiveness and efficiency. Code is available at
https://github.com/wyhallenwu/how2compress and a reproducible docker image at
https://hub.docker.com/r/wuyuheng/how2compress.

### 4. [On AI Verification in Open RAN](http://arxiv.org/pdf/2510.18417v1)

Authors: Rahul Soundrarajan, Claudio Fiandrino, Michele Polese, Salvatore D'Oro, Leonardo Bonati, Tommaso Melodia

Open RAN introduces a flexible, cloud-based architecture for the Radio Access
Network (RAN), enabling Artificial Intelligence (AI)/Machine Learning
(ML)-driven automation across heterogeneous, multi-vendor deployments. While
EXplainable Artificial Intelligence (XAI) helps mitigate the opacity of AI
models, explainability alone does not guarantee reliable network operations. In
this article, we propose a lightweight verification approach based on
interpretable models to validate the behavior of Deep Reinforcement Learning
(DRL) agents for RAN slicing and scheduling in Open RAN. Specifically, we use
Decision Tree (DT)-based verifiers to perform near-real-time consistency checks
at runtime, which would be otherwise unfeasible with computationally expensive
state-of-the-art verifiers. We analyze the landscape of XAI and AI
verification, propose a scalable architectural integration, and demonstrate
feasibility with a DT-based slice-verifier. We also outline future challenges
to ensure trustworthy AI adoption in Open RAN.

### 5. [Forward to Hell? On the Potentials of Misusing Transparent DNS Forwarders in Reflective Amplification Attacks](http://arxiv.org/pdf/2510.18572v1)

Authors: Maynard Koch, Florian Dolzmann, Thomas C. Schmidt, Matthias Wählisch

The DNS infrastructure is infamous for facilitating reflective amplification
attacks. Various countermeasures such as server shielding, access control, rate
limiting, and protocol restrictions have been implemented. Still, the threat
remains throughout the deployment of DNS servers. In this paper, we report on
and evaluate the often unnoticed threat that derives from transparent DNS
forwarders, a widely deployed, incompletely functional set of DNS components.
Transparent DNS forwarders transfer DNS requests without rebuilding packets
with correct source addresses. As such, transparent forwarders feed DNS
requests into (mainly powerful and anycasted) open recursive resolvers, which
thereby can be misused to participate unwillingly in distributed reflective
amplification attacks. We show how transparent forwarders raise severe threats
to the Internet infrastructure. They easily circumvent rate limiting and
achieve an additional, scalable impact via the DNS anycast infrastructure. We
empirically verify this scaling behavior up to a factor of 14. Transparent
forwarders can also assist in bypassing firewall rules that protect recursive
resolvers, making these shielded infrastructure entities part of the global DNS
attack surface.

### 6. [Formal Methods for Mobile Ad Hoc Networks: A Survey](http://arxiv.org/pdf/2510.18730v1)

Authors: Wan Fokkink, Rob van Glabbeek

In a mobile ad hoc network (MANET), communication is wireless and nodes can
move independently. Properly analyzing the functional correctness, performance,
and security of MANET protocols is a challenging task. A wide range of formal
specification and analysis techniques have been employed in the analysis of
MANET protocols. This survey presents an overview of rigorous formal analysis
techniques and their applications, with a focus on MANET routing protocols.
Next to functional correctness, also real-time properties and security are
considered. Moreover, an overview is given of formal frameworks that target
MANETs specifically, as well as mobility models that underlie performance
analyses of MANET protocols. The aim is to give a comprehensive and coherent
overview of this rather scattered field, in which a variety of rigorous formal
methods have been applied to analyze different aspects of a wide range of MANET
protocols.

### 7. [Censorship Chokepoints: New Battlegrounds for Regional Surveillance, Censorship and Influence on the Internet](http://arxiv.org/pdf/2510.18394v1)

Authors: Yong Zhang, Nishanth Sastry

Undoubtedly, the Internet has become one of the most important conduits to
information for the general public. Nonetheless, Internet access can be and has
been limited systematically or blocked completely during political events in
numerous countries and regions by various censorship mechanisms. Depending on
where the core filtering component is situated, censorship techniques have been
classified as client-based, server-based, or network-based. However, as the
Internet evolves rapidly, new and sophisticated censorship techniques have
emerged, which involve techniques that cut across locations and involve new
forms of hurdles to information access. We argue that modern censorship can be
better understood through a new lens that we term chokepoints, which identifies
bottlenecks in the content production or delivery cycle where efficient new
forms of large-scale client-side surveillance and filtering mechanisms have
emerged.

### 8. [sNVMe-oF: Secure and Efficient Disaggregated Storage](http://arxiv.org/pdf/2510.18756v1)

Authors: Marcin Chrapek, Meni Orenbach, Ahmad Atamli, Marcin Copik, Fritz Alder, Torsten Hoefler

Disaggregated storage with NVMe-over-Fabrics (NVMe-oF) has emerged as the
standard solution in modern data centers, achieving superior performance,
resource utilization, and power efficiency. Simultaneously, confidential
computing (CC) is becoming the de facto security paradigm, enforcing stronger
isolation and protection for sensitive workloads. However, securing
state-of-the-art storage with traditional CC methods struggles to scale and
compromises performance or security. To address these issues, we introduce
sNVMe-oF, a storage management system extending the NVMe-oF protocol and
adhering to the CC threat model by providing confidentiality, integrity, and
freshness guarantees. sNVMe-oF offers an appropriate control path and novel
concepts such as counter-leasing. sNVMe-oF also optimizes data path performance
by leveraging NVMe metadata, introducing a new disaggregated Hazel Merkle Tree
(HMT), and avoiding redundant IPSec protections. We achieve this without
modifying the NVMe-oF protocol. To prevent excessive resource usage while
delivering line rate, sNVMe-oF also uses accelerators of CC-capable smart NICs.
We prototype sNVMe-oF on an NVIDIA BlueField-3 and demonstrate how it can
achieve as little as 2% performance degradation for synthetic patterns and AI
training.

### Robotics

### 1. [MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning](http://arxiv.org/pdf/2510.18337v1)

Authors: Wenhui Huang, Changhe Chen, Han Qi, Chen Lv, Yilun Du, Heng Yang

Integrating visual-language instructions into visuomotor policies is gaining
momentum in robot learning for enhancing open-world generalization. Despite
promising advances, existing approaches face two challenges: limited language
steerability when no generated reasoning is used as a condition, or significant
inference latency when reasoning is incorporated.In this work, we introduce
MoTVLA, a mixture-of-transformers (MoT)-based vision-language-action (VLA)
model that integrates fast-slow unified reasoning with behavior policy
learning. MoTVLA preserves the general intelligence of pre-trained VLMs
(serving as the generalist) for tasks such as perception, scene understanding,
and semantic planning, while incorporating a domain expert, a second
transformer that shares knowledge with the pretrained VLM, to generate
domain-specific fast reasoning (e.g., robot motion decomposition), thereby
improving policy execution efficiency. By conditioning the action expert on
decomposed motion instructions, MoTVLA can learn diverse behaviors and
substantially improve language steerability. Extensive evaluations across
natural language processing benchmarks, robotic simulation environments, and
real-world experiments confirm the superiority of MoTVLA in both fast-slow
reasoning and manipulation task performance.

### 2. [Biomechanically consistent real-time action recognition for human-robot interaction](http://arxiv.org/pdf/2510.18373v1)

Authors: Wanchen Li, Kahina Chalabi, Sabbah Maxime, Thomas Bousquet, Robin Passama, Sofiane Ramdani, Andrea Cherubini, Vincent Bonnet

This paper presents a novel framework for real-time human action recognition
in industrial contexts, using standard 2D cameras. We introduce a complete
pipeline for robust and real-time estimation of human joint kinematics, input
to a temporally smoothed Transformer-based network, for action recognition. We
rely on a new dataset including 11 subjects performing various actions, to
evaluate our approach. Unlike most of the literature that relies on joint
center positions (JCP) and is offline, ours uses biomechanical prior, eg. joint
angles, for fast and robust real-time recognition. Besides, joint angles make
the proposed method agnostic to sensor and subject poses as well as to
anthropometric differences, and ensure robustness across environments and
subjects. Our proposed learning model outperforms the best baseline model,
running also in real-time, along various metrics. It achieves 88% accuracy and
shows great generalization ability, for subjects not facing the cameras.
Finally, we demonstrate the robustness and usefulness of our technique, through
an online interaction experiment, with a simulated robot controlled in
real-time via the recognized actions.

### 3. [Efficient Model-Based Reinforcement Learning for Robot Control via Online Learning](http://arxiv.org/pdf/2510.18518v1)

Authors: Fang Nan, Hao Ma, Qinghua Guan, Josie Hughes, Michael Muehlebach, Marco Hutter

We present an online model-based reinforcement learning algorithm suitable
for controlling complex robotic systems directly in the real world. Unlike
prevailing sim-to-real pipelines that rely on extensive offline simulation and
model-free policy optimization, our method builds a dynamics model from
real-time interaction data and performs policy updates guided by the learned
dynamics model. This efficient model-based reinforcement learning scheme
significantly reduces the number of samples to train control policies, enabling
direct training on real-world rollout data. This significantly reduces the
influence of bias in the simulated data, and facilitates the search for
high-performance control policies. We adopt online learning analysis to derive
sublinear regret bounds under standard stochastic online optimization
assumptions, providing formal guarantees on performance improvement as more
interaction data are collected. Experimental evaluations were performed on a
hydraulic excavator arm and a soft robot arm, where the algorithm demonstrates
strong sample efficiency compared to model-free reinforcement learning methods,
reaching comparable performance within hours. Robust adaptation to shifting
dynamics was also observed when the payload condition was randomized. Our
approach paves the way toward efficient and reliable on-robot learning for a
broad class of challenging control tasks.

### 4. [Flexbee: A Grasping and Perching UAV Based on Soft Vector-Propulsion Nozzle](http://arxiv.org/pdf/2510.18558v1)

Authors: Yue Wang, Lixian Zhang, Yimin Zhu, Yangguang Liu, Xuwei Yang

The aim of this paper is to design a new type of grasping and perching
unmanned aerial vehicle (UAV), called Flexbee, which features a soft
vector-propulsion nozzle (SVPN). Compared to previous UAVs, Flexbee integrates
flight, grasping, and perching functionalities into the four SVPNs. This
integration offers advantages including decoupled position and attitude
control, high structural reuse, and strong adaptability strong adaptability for
grasping and perching. A dynamics model of Flexbee has been developed, and the
nonlinear coupling issue of the moment has been resolved through linearization
of the equivalent moment model. A hierarchical control strategy was used to
design controllers for the two operational modes of Flexbee. Finally, flight,
grasping, and perching experiments were conducted to validate Flexbee's
kinematic capabilities and the effectiveness of the control strategy.

### 5. [Quadrupeds for Planetary Exploration: Field Testing Control Algorithms on an Active Volcano](http://arxiv.org/pdf/2510.18600v1)

Authors: Shubham Vyas, Franek Stark, Rohit Kumar, Hannah Isermann, Jonas Haack, Mihaela Popescu, Jakob Middelberg, Dennis Mronga, Frank Kirchner

Missions such as the Ingenuity helicopter have shown the advantages of using
novel locomotion modes to increase the scientific return of planetary
exploration missions. Legged robots can further expand the reach and capability
of future planetary missions by traversing more difficult terrain than wheeled
rovers, such as jumping over cracks on the ground or traversing rugged terrain
with boulders. To develop and test algorithms for using quadruped robots, the
AAPLE project was carried out at DFKI. As part of the project, we conducted a
series of field experiments on the Volcano on the Aeolian island of Vulcano, an
active stratovolcano near Sicily, Italy. The experiments focused on validating
newly developed state-of-the-art adaptive optimal control algorithms for
quadrupedal locomotion in a high-fidelity analog environment for Lunar and
Martian surfaces. This paper presents the technical approach, test plan,
software architecture, field deployment strategy, and evaluation results from
the Vulcano campaign.

### 6. [Least Restrictive Hyperplane Control Barrier Functions](http://arxiv.org/pdf/2510.18643v1)

Authors: Mattias Trende, Petter Ögren

Control Barrier Functions (CBFs) can provide provable safety guarantees for
dynamic systems. However, finding a valid CBF for a system of interest is often
non-trivial, especially if the shape of the unsafe region is complex and the
CBFs are of higher order. A common solution to this problem is to make a
conservative approximation of the unsafe region in the form of a
line/hyperplane, and use the corresponding conservative Hyperplane-CBF when
deciding on safe control actions. In this letter, we note that conservative
constraints are only a problem if they prevent us from doing what we want.
Thus, instead of first choosing a CBF and then choosing a safe control with
respect to the CBF, we optimize over a combination of CBFs and safe controls to
get as close as possible to our desired control, while still having the safety
guarantee provided by the CBF. We call the corresponding CBF the least
restrictive Hyperplane-CBF. Finally, we also provide a way of creating a smooth
parameterization of the CBF-family for the optimization, and illustrate the
approach on a double integrator dynamical system with acceleration constraints,
moving through a group of arbitrarily shaped static and moving obstacles.

### 7. [Towards An Adaptive Locomotion Strategy For Quadruped Rovers: Quantifying When To Slide Or Walk On Planetary Slopes](http://arxiv.org/pdf/2510.18678v1)

Authors: Alberto Sanchez-Delgado, João Carlos Virgolino Soares, David Omar Al Tawil, Alessia Li Noce, Matteo Villa, Victor Barasuol, Paolo Arena, Claudio Semini

Legged rovers provide enhanced mobility compared to wheeled platforms,
enabling navigation on steep and irregular planetary terrains. However,
traditional legged locomotion might be energetically inefficient and
potentially dangerous to the rover on loose and inclined surfaces, such as
crater walls and cave slopes. This paper introduces a preliminary study that
compares the Cost of Transport (CoT) of walking and torso-based sliding
locomotion for quadruped robots across different slopes, friction conditions
and speed levels. By identifying intersections between walking and sliding CoT
curves, we aim to define threshold conditions that may trigger transitions
between the two strategies. The methodology combines physics-based simulations
in Isaac Sim with particle interaction validation in ANSYS-Rocky. Our results
represent an initial step towards adaptive locomotion strategies for planetary
legged rovers.

### 8. [Event-Grounding Graph: Unified Spatio-Temporal Scene Graph from Robotic Observations](http://arxiv.org/pdf/2510.18697v1)

Authors: Phuoc Nguyen, Francesco Verdoja, Ville Kyrki

A fundamental aspect for building intelligent autonomous robots that can
assist humans in their daily lives is the construction of rich environmental
representations. While advances in semantic scene representations have enriched
robotic scene understanding, current approaches lack a connection between
spatial features and dynamic events; e.g., connecting the blue mug to the event
washing a mug. In this work, we introduce the event-grounding graph (EGG), a
framework grounding event interactions to spatial features of a scene. This
representation allows robots to perceive, reason, and respond to complex
spatio-temporal queries. Experiments using real robotic data demonstrate EGG's
capability to retrieve relevant information and respond accurately to human
inquiries concerning the environment and events within. Furthermore, the EGG
framework's source code and evaluation dataset are released as open-source at:
https://github.com/aalto-intelligent-robotics/EGG.

### 9. [Sharing the Load: Distributed Model-Predictive Control for Precise Multi-Rover Cargo Transport](http://arxiv.org/pdf/2510.18766v1)

Authors: Alexander Krawciw, Sven Lilge, Luka Antonyshyn, Timothy D. Barfoot

For autonomous cargo transportation, teams of mobile robots can provide more
operational flexibility than a single large robot. In these scenarios,
precision in both inter-vehicle distance and path tracking is key. With this
motivation, we develop a distributed model-predictive controller (MPC) for
multi-vehicle cargo operations that builds on the precise path-tracking of
lidar teach and repeat. To carry cargo, a following vehicle must maintain a
Euclidean distance offset from a lead vehicle regardless of the path curvature.
Our approach uses a shared map to localize the robots relative to each other
without GNSS or direct observations. We compare our approach to a centralized
MPC and a baseline approach that directly measures the inter-vehicle distance.
The distributed MPC shows equivalent nominal performance to the more complex
centralized MPC. Using a direct measurement of the relative distance between
the leader and follower shows improved tracking performance in close-range
scenarios but struggles with long-range offsets. The operational flexibility
provided by distributing the computation makes it well suited for real
deployments. We evaluate four types of convoyed path trackers with over 10 km
of driving in a coupled convoy. With convoys of two and three rovers, the
proposed distributed MPC method works in real-time to allow map-based convoying
to maintain maximum spacing within 20 cm of the target in various conditions.

### 10. [Online Object-Level Semantic Mapping for Quadrupeds in Real-World Environments](http://arxiv.org/pdf/2510.18776v1)

Authors: Emad Razavi, Angelo Bratta, João Carlos Virgolino Soares, Carmine Recchiuto, Claudio Semini

We present an online semantic object mapping system for a quadruped robot
operating in real indoor environments, turning sensor detections into named
objects in a global map. During a run, the mapper integrates range geometry
with camera detections, merges co-located detections within a frame, and
associates repeated detections into persistent object instances across frames.
Objects remain in the map when they are out of view, and repeated sightings
update the same instance rather than creating duplicates. The output is a
compact object layer that can be queried (class, pose, and confidence), is
integrated with the occupancy map and readable by a planner. In on-robot tests,
the layer remained stable across viewpoint changes.

### Software Engineering

### 1. [When Old Meets New: Evaluating the Impact of Regression Tests on SWE Issue Resolution](http://arxiv.org/pdf/2510.18270v1)

Authors: Yang Chen, Toufique Ahmed, Reyhaneh Jabbarvand, Martin Hirzel

Test suites in real-world projects are often large and achieve high code
coverage, yet they remain insufficient for detecting all bugs. The abundance of
unresolved issues in open-source project trackers highlights this gap. While
regression tests are typically designed to ensure past functionality is
preserved in the new version, they can also serve a complementary purpose:
debugging the current version. Specifically, regression tests can (1) enhance
the generation of reproduction tests for newly reported issues, and (2)
validate that patches do not regress existing functionality. We present
TestPrune, a fully automated technique that leverages issue tracker reports and
strategically reuses regression tests for both bug reproduction and patch
validation.
  A key contribution of TestPrune is its ability to automatically minimize the
regression suite to a small, highly relevant subset of tests. Due to the
predominance of LLM-based debugging techniques, this minimization is essential
as large test suites exceed context limits, introduce noise, and inflate
inference costs. TestPrune can be plugged into any agentic bug repair pipeline
and orthogonally improve overall performance. As a proof of concept, we show
that TestPrune leads to a 6.2%-9.0% relative increase in issue reproduction
rate within the Otter framework and a 9.4% - 12.9% relative increase in issue
resolution rate within the Agentless framework on SWE-Bench Lite and SWE-Bench
Verified benchmarks, capturing fixes that were correctly produced by agents but
not submitted as final patches. Compared to the benefits, the cost overhead of
using TestPrune is minimal, i.e., \$0.02 and \$0.05 per SWE-Bench instance,
using GPT-4o and Claude-3.7-Sonnet models, respectively.

### 2. [Ensuring Robustness in ML-enabled Software Systems: A User Survey](http://arxiv.org/pdf/2510.18292v1)

Authors: Hala Abdelkader, Mohamed Abdelrazek, Priya Rani, Rajesh Vasa, Jean-Guy Schneider

Ensuring robustness in ML-enabled software systems requires addressing
critical challenges, such as silent failures, out-of-distribution (OOD) data,
and adversarial attacks. Traditional software engineering practices, which rely
on predefined logic, are insufficient for ML components that depend on data and
probabilistic decision-making. To address these challenges, we propose the
ML-On-Rails protocol, a unified framework designed to enhance the robustness
and trustworthiness of ML-enabled systems in production. This protocol
integrates key safeguards such as OOD detection, adversarial attack detection,
input validation, and explainability. It also includes a model-to-software
communication framework using HTTP status codes to enhance transparency in
reporting model outcomes and errors. To align our approach with real-world
challenges, we conducted a practitioner survey, which revealed major robustness
issues, gaps in current solutions, and highlighted how a standardised protocol
such as ML-On-Rails can improve system robustness. Our findings highlight the
need for more support and resources for engineers working with ML systems.
Finally, we outline future directions for refining the proposed protocol,
leveraging insights from the survey and real-world applications to continually
enhance its effectiveness.

### 3. [InspectCoder: Dynamic Analysis-Enabled Self Repair through interactive LLM-Debugger Collaboration](http://arxiv.org/pdf/2510.18327v1)

Authors: Yunkun Wang, Yue Zhang, Guochang Li, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, Shuiguang Deng

Large Language Models (LLMs) frequently generate buggy code with complex
logic errors that are challenging to diagnose. While existing LLM-based
self-repair approaches conduct intensive static semantic analysis or reply on
superficial execution logs, they miss the in-depth runtime behaviors that often
expose bug root causes-lacking the interactive dynamic analysis capabilities
that make human debugging effective. We present InspectCoder, the first agentic
program repair system that empowers LLMs to actively conduct dynamic analysis
via interactive debugger control. Our dual-agent framework enables strategic
breakpoint placement, targeted state inspection, and incremental runtime
experimentation within stateful debugger sessions. Unlike existing methods that
follow fixed log collection procedures, InspectCoder adaptively inspects and
perturbs relevant intermediate states at runtime, and leverages immediate
process rewards from debugger feedback to guide multi-step reasoning,
transforming LLM debugging paradigm from blind trial-and-error into systematic
root cause diagnosis. We conduct comprehensive experiments on two challenging
self-repair benchmarks: BigCodeBench-R and LiveCodeBench-R. InspectCoder
achieves 5.10%-60.37% relative improvements in repair accuracy over the
strongest baseline, while delivering 1.67x-2.24x superior bug-fix efficiency
respectively. We also contribute InspectWare, an open-source middleware that
abstracts debugger complexities and maintains stateful debugging sessions
across mainstream Python testing frameworks. Our work provides actionable
insight into the interactive LLM-debugger systems, demonstrating the
significant potential of LLM-driven dynamic analysis for automated software
engineering.

### 4. [Human to Document, AI to Code: Three Case Studies of Comparing GenAI for Notebook Competitions](http://arxiv.org/pdf/2510.18430v1)

Authors: Tasha Settewong, Youmei Fan, Raula Gaikovina Kula, Kenichi Matsumoto

Computational notebooks have become the preferred tool of choice for data
scientists and practitioners to perform analyses and share results. Notebooks
uniquely combine scripts with documentation. With the emergence of generative
AI (GenAI) technologies, it is increasingly important, especially in
competitive settings, to distinguish the characteristics of human-written
versus GenAI.
  In this study, we present three case studies to explore potential strengths
of both humans and GenAI through the coding and documenting activities in
notebooks. We first characterize differences between 25 code and documentation
features in human-written, medal-winning Kaggle notebooks. We find that gold
medalists are primarily distinguished by longer and more detailed
documentation. Second, we analyze the distinctions between human-written and
GenAI notebooks. Our results show that while GenAI notebooks tend to achieve
higher code quality (as measured by metrics like code smells and technical
debt), human-written notebooks display greater structural diversity,
complexity, and innovative approaches to problem-solving. Based on these
results, we envision the work as groundwork that highlight four agendas to
further investigate how GenAI could be utilized in notebooks that maximizes the
potential collaboration between human and AI.

### 5. [Large Language Models in Thematic Analysis: Prompt Engineering, Evaluation, and Guidelines for Qualitative Software Engineering Research](http://arxiv.org/pdf/2510.18456v1)

Authors: Cristina Martinez Montes, Robert Feldt, Cristina Miguel Martos, Sofia Ouhbi, Shweta Premanandan, Daniel Graziotin

As artificial intelligence advances, large language models (LLMs) are
entering qualitative research workflows, yet no reproducible methods exist for
integrating them into established approaches like thematic analysis (TA), one
of the most common qualitative methods in software engineering research.
Moreover, existing studies lack systematic evaluation of LLM-generated
qualitative outputs against established quality criteria. We designed and
iteratively refined prompts for Phases 2-5 of Braun and Clarke's reflexive TA,
then tested outputs from multiple LLMs against codes and themes produced by
experienced researchers. Using 15 interviews on software engineers' well-being,
we conducted blind evaluations with four expert evaluators who applied rubrics
derived directly from Braun and Clarke's quality criteria. Evaluators preferred
LLM-generated codes 61% of the time, finding them analytically useful for
answering the research question. However, evaluators also identified
limitations: LLMs fragmented data unnecessarily, missed latent interpretations,
and sometimes produced themes with unclear boundaries. Our contributions are
threefold. First, a reproducible approach integrating refined, documented
prompts with an evaluation framework to operationalize Braun and Clarke's
reflexive TA. Second, an empirical comparison of LLM- and human-generated codes
and themes in software engineering data. Third, guidelines for integrating LLMs
into qualitative analysis while preserving methodological rigour, clarifying
when and how LLMs can assist effectively and when human interpretation remains
essential.

### 6. [VAPU: System for Autonomous Legacy Code Modernization](http://arxiv.org/pdf/2510.18509v1)

Authors: Valtteri Ala-Salmi, Zeeshan Rasheed, Abdul Malik Sami, Muhammad Waseem, Kai-Kristian Kemell, Jussi Rasku, Mika Saari, Pekka Abrahamsson

In this study, we present a solution for the modernization of legacy
applications, an area of code generation where LLM-based multi-agent systems
are proving essential for complex multi-phased tasks. Legacy applications often
contain deprecated components that create compatibility, security, and
reliability risks, but high resource costs make companies hesitate to update.
We take a step forward to integrate an LLM-based multi-agent system as part of
a legacy web application update to provide a cost-effective solution to update
legacy applications autonomously. We propose a multi-agent system named a
Verifying Agent Pipeline Updater (VAPU), which is designed to update code files
in phases while simulating different roles in a software development team. In
our previous study, we evaluated the system for legacy version updates by using
six legacy web application view files by resulting errors and accomplished
requirements. This study extends the previous evaluation of a multi-agent
pipeline system by extending the evaluation of VAPU from a single LLM to five
LLMs and using the temperature parameter in both 0 to 1 settings. Additionally,
we tested the system with 20 open-source Python GitHub projects. The results of
the evaluation were compared to Zero-Shot Learning (ZSL) and One-Shot Learning
(OSL) prompts. The extended evaluation of VAPU showed that particularly in a
low-temperature VAPU can get similar level of error count compared to the
ZSL/OSL prompts but with a higher level of fulfilled requirements, depending on
the LLM. VAPU showed up to 22.5% increase in the succeeding Python file update
requirements compared to ZSL/OSL prompts. The study indicates that an LLM-based
multi-agent system is a capable solution to update components of a legacy
application autonomously.

### 7. [Mining Service Behavior for Stateful Service Emulation](http://arxiv.org/pdf/2510.18519v1)

Authors: Md Arafat Hossain, Jun Han, Muhammad Ashad Kabir, Steve Versteeg, Jean-Guy Schneider, Jiaojiao Jiang

Enterprise software systems are increasingly integrating with diverse
services to meet expanding business demands. Testing these highly
interconnected systems presents a challenge due to the need for access to the
connected services. Service virtualization has emerged as a widely used
technique to derive service models from recorded interactions, for service
response generation during system testing. Various methods have been proposed
to emulate actual service behavior based on these interactions, but most fail
to account for the service's state, which reduces the accuracy of service
emulation and the realism of the testing environment, especially when dealing
with stateful services. This paper proposes an approach to deriving service
models from service interactions, which enhance the accuracy of response
generation by considering service state. This is achieved by uncovering
contextual dependencies among interaction messages and analyzing the
relationships between message data values. The approach is evaluated using
interaction traces collected from both stateful and stateless services, and the
results reveal notable enhancements in accuracy and efficiency over existing
approaches in service response generation.

### 8. [Demonstrators for Industrial Cyber-Physical System Research: A Requirements Hierarchy Driven by Software-Intensive Design](http://arxiv.org/pdf/2510.18534v1)

Authors: Uraz Odyurt, Richard Loendersloot, Tiedo Tinga

One of the challenges apparent in the organisation of research projects is
the uncertainties around the subject of demonstrators. A precise and detailed
elicitation of the coverage for project demonstrators is often an afterthought
and not sufficiently detailed during proposal writing. This practice leads to
continuous confusion and a mismatch between targeted and achievable
demonstration of results, hindering progress. The reliance on the TRL scale as
a loose descriptor does not help either. We propose a demonstrator requirements
elaboration framework aiming to evaluate the feasibility of targeted
demonstrations, making realistic adjustments, and assist in describing
requirements. In doing so, we define 5 hierarchical levels of demonstration,
clearly connected to expectations, e.g., work package interaction, and also
connected to the project's industrial use-cases. The considered application
scope in this paper is the domain of software-intensive systems and industrial
cyber-physical systems. A complete validation is not accessible, as it would
require application of our framework at the start of a project and observing
the results at the end, taking 4-5 years. Nonetheless, we have applied it to
two research projects from our portfolio, one at the early and another at the
final stages, revealing its effectiveness.

### 9. [An overview of the use of alternative funding and contracting approaches relevant for agile software development: A systematic review of real-life experiences](http://arxiv.org/pdf/2510.18711v1)

Authors: Bertha Ngereja, Magne Jørgensen

Agile software development emphasizes flexibility and iterative processes,
which may conflict with the more linear, rigid, and time-consuming traditional
funding and contracting approaches. This review synthesizes real-life
experiences of using alternative (non-traditional) contracting and funding
approaches. The focus is on identifying approaches that align better with agile
principles and understanding the motivations, benefits, and challenges these
alternatives present. A systematic literature review was conducted in SCOPUS,
Web of Science, and Google Scholar, where we identified 38 relevant
peer-reviewed empirical studies from private and public sector contexts. Four
alternative funding and four alternative contracting approaches were
identified. Organizations were motivated to adopt these alternative approaches
because traditional approaches often proved too rigid, conflicted with agile
principles, hindered effective client-contractor collaboration, and limited
profitability. The benefits of these alternatives included higher client
satisfaction, reduced contractor risk, and more efficient resource utilization.
Adopting alternative funding and contracting approaches may promote flexibility
and efficiency in agile projects but also presents cultural and structural
challenges, increases the risk of scope creep and analysis paralysis, and
requires additional effort in terms of time and resources. The context of the
organization matters highly in selecting a suitable approach, such as the
organizational readiness in terms of its leaders, people, and systems. Thus,
instead of wholly adopting alternative approaches and introducing changes
abruptly, organizations may benefit from starting with hybrid approaches that
balance flexibility and control and progressively transition to fully flexible
approaches tailored to their needs

### 10. [ShaRE your Data! Characterizing Datasets for LLM-based Requirements Engineering](http://arxiv.org/pdf/2510.18787v1)

Authors: Quim Motger, Carlota Catot, Xavier Franch

[Context] Large Language Models (LLMs) rely on domain-specific datasets to
achieve robust performance across training and inference stages. However, in
Requirements Engineering (RE), data scarcity remains a persistent limitation
reported in surveys and mapping studies. [Question/Problem] Although there are
multiple datasets supporting LLM-based RE tasks (LLM4RE), they are fragmented
and poorly characterized, limiting reuse and comparability. This research
addresses the limited visibility and characterization of datasets used in
LLM4RE. We investigate which public datasets are employed, how they can be
systematically characterized, and which RE tasks and dataset descriptors remain
under-represented. [Ideas/Results] To address this, we conduct a systematic
mapping study to identify and analyse datasets used in LLM4RE research. A total
of 62 publicly available datasets are referenced across 43 primary studies.
Each dataset is characterized along descriptors such as artifact type,
granularity, RE stage, task, domain, and language. Preliminary findings show
multiple research gaps, including limited coverage for elicitation tasks,
scarce datasets for management activities beyond traceability, and limited
multilingual availability. [Contribution] This research preview offers a public
catalogue and structured characterization scheme to support dataset selection,
comparison, and reuse in LLM4RE research. Future work will extend the scope to
grey literature, as well as integration with open dataset and benchmark
repositories.

### Social and Information Networks

### 1. [Multiplex Networks Provide Structural Pathways for Social Contagion in Rural Social Networks](http://arxiv.org/pdf/2510.18280v1)

Authors: Yongren Shi, Edo Airoldi, Nicholas A. Christakis

Human social networks are inherently multiplex, comprising overlapping layers
of relationships. Different layers may have distinct structural properties and
interpersonal dynamics, but also may interact to form complex interdependent
pathways for social contagion. This poses a fundamental problem in
understanding behavioral diffusion and in devising effective network-based
interventions. Here, we introduce a new conceptualization of how much each
network layer contributes to critical contagion pathways and quantify it using
a novel metric, network torque. We exploit data regarding sociocentric maps of
110 rural Honduran communities using a battery of 11 name generators and an
experiment involving an exogenous intervention. Using a novel statistical
framework, we assess the extent to which specific network layers alter global
connectivity and support the spread of three experimentally introduced health
practices. The results show that specific relationship types - such as close
friendships - particularly enable non-overlapping diffusion pathways,
amplifying behavioral change at the village level. For instance, non-redundant
pathways enabled by closest friends can increase the adoption of correct
knowledge about feeding newborns inappropriate chupones and enhance attitudes
regarding fathers' involvement in postpartum care. Non-overlapping multiplex
social ties are relevant to social contagion and social coherence in
traditionally organized social systems.

### 2. [MoveOD: Synthesizing Origin-Destination Commute Distribution from U.S. Census Data](http://arxiv.org/pdf/2510.18858v1)

Authors: Rishav Sen, Abhishek Dubey, Ayan Mukhopadhyay, Samitha Samaranayake, Aron Laszka

High-resolution origin-destination (OD) tables are essential for a wide
spectrum of transportation applications, from modeling traffic and signal
timing optimization to congestion pricing and vehicle routing. However, outside
a handful of data rich cities, such data is rarely available. We introduce
MOVEOD, an open-source pipeline that synthesizes public data into commuter OD
flows with fine-grained spatial and temporal departure times for any county in
the United States. MOVEOD combines five open data sources: American Community
Survey (ACS) departure time and travel time distributions, Longitudinal
Employer-Household Dynamics (LODES) residence-to-workplace flows, county
geometries, road network information from OpenStreetMap (OSM), and building
footprints from OSM and Microsoft, into a single OD dataset. We use a
constrained sampling and integer-programming method to reconcile the OD dataset
with data from ACS and LODES. Our approach involves: (1) matching commuter
totals per origin zone, (2) aligning workplace destinations with employment
distributions, and (3) calibrating travel durations to ACS-reported commute
times. This ensures the OD data accurately reflects commuting patterns. We
demonstrate the framework on Hamilton County, Tennessee, where we generate
roughly 150,000 synthetic trips in minutes, which we feed into a benchmark
suite of classical and learning-based vehicle-routing algorithms. The MOVEOD
pipeline is an end-to-end automated system, enabling users to easily apply it
across the United States by giving only a county and a year; and it can be
adapted to other countries with comparable census datasets. The source code and
a lightweight browser interface are publicly available.

### 3. [Censorship Chokepoints: New Battlegrounds for Regional Surveillance, Censorship and Influence on the Internet](http://arxiv.org/pdf/2510.18394v1)

Authors: Yong Zhang, Nishanth Sastry

Undoubtedly, the Internet has become one of the most important conduits to
information for the general public. Nonetheless, Internet access can be and has
been limited systematically or blocked completely during political events in
numerous countries and regions by various censorship mechanisms. Depending on
where the core filtering component is situated, censorship techniques have been
classified as client-based, server-based, or network-based. However, as the
Internet evolves rapidly, new and sophisticated censorship techniques have
emerged, which involve techniques that cut across locations and involve new
forms of hurdles to information access. We argue that modern censorship can be
better understood through a new lens that we term chokepoints, which identifies
bottlenecks in the content production or delivery cycle where efficient new
forms of large-scale client-side surveillance and filtering mechanisms have
emerged.

### Systems and Control

### 1. [Urban Air Mobility: A Review of Recent Advances in Communication, Management, and Sustainability](http://arxiv.org/pdf/2510.18235v1)

Authors: Zhitong He, Zijing Wang, Lingxi Li

Urban Air Mobility (UAM) offers a transformative approach to addressing urban
congestion, improving accessibility, and advancing environmental
sustainability. Rapid progress has emerged in three tightly linked domains
since 2020: (1) Communication, where dynamic spectrum allocation and
low-altitude channel characterization support reliable air-ground data
exchange; (2) UAM management, with novel air-traffic control concepts for
dense, largely autonomous urban airspace; and (3) Sustainability, driven by
energy-efficient propulsion, integrated charging infrastructure, and holistic
environmental assessment. This paper reviews and synthesizes the latest
research across these areas, compares the state-of-the-art solutions, and
outlines the technological and infrastructural milestones that are critical to
realizing a scalable, sustainable UAM ecosystem.

### 2. [Sliding-Mode Control Strategies for PMSM speed control: A Comprehensive Review, Taxonomy and Research Gaps](http://arxiv.org/pdf/2510.18420v1)

Authors: Abdullah Ajasa, Mubarak Badamasi Aremu, Ali Nasir

Permanent Magnet Synchronous Motors (PMSMs) are widely employed in
high-performance drive systems due to their high efficiency, power density, and
precise dynamic behavior. However, nonlinearities, load disturbances, and
parameter uncertainties present persistent challenges to control. Sliding-Mode
Control (SMC) remains one of the most reliable strategies for high-performance
PMSM drives. Yet, the rapid proliferation of adaptive, fractional-order, and
intelligent variants has fragmented recent literature. This paper presents a
comprehensive review and taxonomy of SMC-based PMSM speed-control methods
published between 2020 and 2025. More than 200 studies are systematically
analyzed and classified according to control order, surface design,
disturbance-observer integration, optimization approach, and intelligent
augmentation. Trends in publication activity, dominant hybrid structures, and
application domains are quantitatively summarized. The review reveals a clear
evolution from conventional discontinuous SMC toward adaptive, higher-order,
and data-driven frameworks that mitigate chattering while preserving
robustness. Persistent research gaps are identified in hardware validation,
energy-efficiency assessment, and real-time tuning strategies. The taxonomy and
critical synthesis provided herein establish a coherent reference for
researchers and form the conceptual foundation for the companion paper (Part
II), which delivers a unified benchmark and comparative simulation study of
representative SMC designs.

### 3. [$\ell_1$-Based Adaptive Identification under Quantized Observations with Applications](http://arxiv.org/pdf/2510.18738v1)

Authors: Xin Zheng, Yifei Jin, Yujing Liu, Lei Guo

Quantized observations are ubiquitous in a wide range of applications across
engineering and the social sciences, and algorithms based on the $\ell_1$-norm
are well recognized for their robustness to outliers compared with their
$\ell_2$-based counterparts. Nevertheless, adaptive identification methods that
integrate quantized observations with $\ell_1$-optimization remain largely
underexplored. Motivated by this gap, we develop a novel $\ell_1$-based
adaptive identification algorithm specifically designed for quantized
observations. Without relying on the traditional persistent excitation
condition, we establish global convergence of the parameter estimates to their
true values and show that the average regret asymptotically vanishes as the
data size increases. Finally, we apply our new identification algorithm to a
judicial sentencing problem using real-world data, which demonstrates its
superior performance and practical significance.

### 4. [Harmonic Cancellation in Multi-Electrolyzer P2H Plants via Phasor-Modulated Production Scheduling](http://arxiv.org/pdf/2510.18223v1)

Authors: Yangjun Zeng, Yiwei Qiu, Li Jiang, Jie Zhu, Yi Zhou, Jiarong Li, Shi Chen, Buxiang Zhou

Thyristor rectifiers (TRs) are cost-effective power supplies for hydrogen
electrolyzers (ELZs) but introduce harmonic distortion that may violate grid
codes. This letter proposes a self-governing harmonic mitigation strategy
through coordinated operation of multiple ELZs in large power-to-hydrogen (P2H)
plants. First, the harmonic model of TR-powered ELZs is derived, revealing a
natural harmonic cancellation mechanism among them. Based on this, a
system-level operation scheme based on phasor modulation is developed and
integrated into plant scheduling. Case studies demonstrate that the proposed
method reduces harmonic currents by 21.2%-39.7% and ensures grid-code
compliance, with only a 0.25% loss in hydrogen output, while increasing total
revenue by over 21\% compared to production-oriented strategies.

### 5. [Brute-force search and Warshall algorithms for matrix-weighted graphs](http://arxiv.org/pdf/2510.18260v1)

Authors: Minh Hoang Trinh, Hyo-Sung Ahn

Although research on the control of networked systems has grown considerably,
graph-theoretic and algorithmic studies on matrix-weighted graphs remain
limited. To bridge this gap in the literature, this work introduces two
algorithms-the brute-force search and the Warshall algorithm-for determining
connectedness and clustering in undirected matrix-weighted graphs. The proposed
algorithms, which are derived from a sufficient condition for connectedness,
emphasize a key distinction between matrix-weighted and scalar-weighted graphs.
While the existence of a path between two vertices guarantees connectedness in
scalar-weighted graphs, connectedness in matrix-weighted graphs is a collective
contribution of all paths joining the two vertices. Proofs of correctness and
numerical examples are provided to illustrate and demonstrate the effectiveness
of the algorithms.

### 6. [Explicit Reformulation of Discrete Distributionally Robust Optimization Problems](http://arxiv.org/pdf/2510.18302v1)

Authors: Yuma Shida, Yuji Ito

Distributionally robust optimization (DRO) is an effective framework for
controlling real-world systems with various uncertainties, typically modeled
using distributional uncertainty balls. However, DRO problems often involve
infinitely many inequality constraints, rendering exact solutions
computationally expensive. In this study, we propose a discrete DRO (DDRO)
method that significantly simplifies the problem by reducing it to a single
trivial constraint. Specifically, the proposed method utilizes two types of
distributional uncertainty balls to reformulate the DDRO problem into a
single-layer smooth convex program, significantly improving tractability.
Furthermore, we provide practical guidance for selecting the appropriate ball
sizes. The original DDRO problem is further reformulated into two optimization
problems: one minimizing the mean and standard deviation, and the other
minimizing the conditional value at risk (CVaR). These formulations account for
the choice of ball sizes, thereby enhancing the practical applicability of the
method. The proposed method was applied to a distributionally robust
patrol-agent design problem, identifying a Pareto front in which the mean and
standard deviation of the mean hitting time varied by up to 3% and 14%,
respectively, while achieving a CVaR reduction of up to 13%.

### 7. [Coverage-Recon: Coordinated Multi-Drone Image Sampling with Online Map Feedback](http://arxiv.org/pdf/2510.18347v1)

Authors: Muhammad Hanif, Reiji Terunuma, Takumi Sumino, Kelvin Cheng, Takeshi Hatanaka

This article addresses collaborative 3D map reconstruction using multiple
drones. Achieving high-quality reconstruction requires capturing images of
keypoints within the target scene from diverse viewing angles, and coverage
control offers an effective framework to meet this requirement. Meanwhile,
recent advances in real-time 3D reconstruction algorithms make it possible to
render an evolving map during flight, enabling immediate feedback to guide
drone motion. Building on this, we present Coverage-Recon, a novel coordinated
image sampling algorithm that integrates online map feedback to improve
reconstruction quality on-the-fly. In Coverage-Recon, the coordinated motion of
drones is governed by a Quadratic Programming (QP)-based angle-aware coverage
controller, which ensures multi-viewpoint image capture while enforcing safety
constraints. The captured images are processed in real time by the NeuralRecon
algorithm to generate an evolving 3D mesh. Mesh changes across the scene are
interpreted as indicators of reconstruction uncertainty and serve as feedback
to update the importance index of the coverage control as the map evolves. The
effectiveness of Coverage-Recon is validated through simulation and
experiments, demonstrating both qualitatively and quantitatively that
incorporating online map feedback yields more complete and accurate 3D
reconstructions than conventional methods. Project page:
https://htnk-lab.github.io/coverage-recon/

### 8. [MMRHP: A Miniature Mixed-Reality HIL Platform for Auditable Closed-Loop Evaluation](http://arxiv.org/pdf/2510.18371v1)

Authors: Mingxin Li, Haibo Hu, Jinghuai Deng, Yuchen Xi, Xinhong Chen, Jianping Wang

Validation of autonomous driving systems requires a trade-off between test
fidelity, cost, and scalability. While miniaturized hardware-in-the-loop (HIL)
platforms have emerged as a promising solution, a systematic framework
supporting rigorous quantitative analysis is generally lacking, limiting their
value as scientific evaluation tools. To address this challenge, we propose
MMRHP, a miniature mixed-reality HIL platform that elevates miniaturized
testing from functional demonstration to rigorous, reproducible quantitative
analysis. The core contributions are threefold. First, we propose a systematic
three-phase testing process oriented toward the Safety of the Intended
Functionality(SOTIF)standard, providing actionable guidance for identifying the
performance limits and triggering conditions of otherwise correctly functioning
systems. Second, we design and implement a HIL platform centered around a
unified spatiotemporal measurement core to support this process, ensuring
consistent and traceable quantification of physical motion and system timing.
Finally, we demonstrate the effectiveness of this solution through
comprehensive experiments. The platform itself was first validated, achieving a
spatial accuracy of 10.27 mm RMSE and a stable closed-loop latency baseline of
approximately 45 ms. Subsequently, an in-depth Autoware case study leveraged
this validated platform to quantify its performance baseline and identify a
critical performance cliff at an injected latency of 40 ms. This work shows
that a structured process, combined with a platform offering a unified
spatio-temporal benchmark, enables reproducible, interpretable, and
quantitative closed-loop evaluation of autonomous driving systems.

### 9. [MPC-based motion planning for non-holonomic systems in non-convex domains](http://arxiv.org/pdf/2510.18402v1)

Authors: Matthias Lorenzen, Teodoro Alamo, Martina Mammarella, Fabrizio Dabbene

Motivated by the application of using model predictive control (MPC) for
motion planning of autonomous mobile robots, a form of output tracking MPC for
non-holonomic systems and with non-convex constraints is studied. Although the
advantages of using MPC for motion planning have been demonstrated in several
papers, in most of the available fundamental literature on output tracking MPC
it is assumed, often implicitly, that the model is holonomic and generally the
state or output constraints must be convex. Thus, in application-oriented
publications, empirical results dominate and the topic of proving completeness,
in particular under which assumptions the target is always reached, has
received comparatively little attention. To address this gap, we present a
novel MPC formulation that guarantees convergence to the desired target under
realistic assumptions, which can be verified in relevant real-world scenarios.

### 10. [Designing trajectories in the Earth-Moon system: a Levenberg-Marquardt approach](http://arxiv.org/pdf/2510.18474v1)

Authors: António Nunes, Sérgio Brás, Pedro Batista, João Xavier

Trajectory design in cislunar space under a High-Fidelity Ephemeris Model
(HFEM) is pursued through a nonlinear optimization perspective anchored on the
transition of solutions from lower fidelity models, namely the Circular
Restricted Three-Body Problem (CR3BP). The optimization problem is posed in the
likeness of a multiple-shooting approach, aiming for segment-to-segment
continuity while tracking proximity to the original CR3BP structures. The
analysis of various formulations leads to the selection of an unconstrained
least-squares problem for further investigation. The nonlinear optimization
problem is convexified and the use of the Levenberg-Marquardt algorithm, as an
alternative to the minimum-norm update equation found in most literature, is
investigated for its control over the update step and inherent robustness.
Additional techniques such as adaptive weighting are employed to further
consolidate the behavior of the proposed algorithm in challenging scenarios.
Numerical trials evaluate the adequacy of the methodology presented and compare
it to the minimum-norm baseline over various application cases, including the
generation of quasi-periodic trajectories and orbital transfers between them.
The proposed approach is found to outperform the baseline in applications where
the initial guess is poor and the ease of including proximity constraints
provides benefits in control over the shape of the converged solution.

### Machine Learning (Statistics Category)

### 1. [The Bias-Variance Tradeoff in Data-Driven Optimization: A Local Misspecification Perspective](http://arxiv.org/pdf/2510.18215v1)

Authors: Haixiang Lan, Luofeng Liao, Adam N. Elmachtoub, Christian Kroer, Henry Lam, Haofeng Zhang

Data-driven stochastic optimization is ubiquitous in machine learning and
operational decision-making problems. Sample average approximation (SAA) and
model-based approaches such as estimate-then-optimize (ETO) or integrated
estimation-optimization (IEO) are all popular, with model-based approaches
being able to circumvent some of the issues with SAA in complex
context-dependent problems. Yet the relative performance of these methods is
poorly understood, with most results confined to the dichotomous cases of the
model-based approach being either well-specified or misspecified. We develop
the first results that allow for a more granular analysis of the relative
performance of these methods under a local misspecification setting, which
models the scenario where the model-based approach is nearly well-specified. By
leveraging tools from contiguity theory in statistics, we show that there is a
bias-variance tradeoff between SAA, IEO, and ETO under local misspecification,
and that the relative importance of the bias and the variance depends on the
degree of local misspecification. Moreover, we derive explicit expressions for
the decision bias, which allows us to characterize (un)impactful
misspecification directions, and provide further geometric understanding of the
variance.

### 2. [Uncertainty Estimation by Flexible Evidential Deep Learning](http://arxiv.org/pdf/2510.18322v1)

Authors: Taeseong Yoon, Heeyoung Kim

Uncertainty quantification (UQ) is crucial for deploying machine learning
models in high-stakes applications, where overconfident predictions can lead to
serious consequences. An effective UQ method must balance computational
efficiency with the ability to generalize across diverse scenarios. Evidential
deep learning (EDL) achieves efficiency by modeling uncertainty through the
prediction of a Dirichlet distribution over class probabilities. However, the
restrictive assumption of Dirichlet-distributed class probabilities limits
EDL's robustness, particularly in complex or unforeseen situations. To address
this, we propose \textit{flexible evidential deep learning}
($\mathcal{F}$-EDL), which extends EDL by predicting a flexible Dirichlet
distribution -- a generalization of the Dirichlet distribution -- over class
probabilities. This approach provides a more expressive and adaptive
representation of uncertainty, significantly enhancing UQ generalization and
reliability under challenging scenarios. We theoretically establish several
advantages of $\mathcal{F}$-EDL and empirically demonstrate its
state-of-the-art UQ performance across diverse evaluation settings, including
classical, long-tailed, and noisy in-distribution scenarios.

### 3. [Parametrising the Inhomogeneity Inducing Capacity of a Training Set, and its Impact on Supervised Learning](http://arxiv.org/pdf/2510.18332v1)

Authors: Gargi Roy, Dalia Chakrabarty

We introduce parametrisation of that property of the available
  training dataset, that necessitates an inhomogeneous correlation
  structure for the function that is learnt as a model of the
  relationship between the pair of variables, observations of which
  comprise the considered training data. We refer to a parametrisation
  of this property of a given training set, as its ``inhomogeneity
  parameter''. It is easy to compute this parameter for small-to-large
  datasets, and we demonstrate such computation on multiple
  publicly-available datasets, while also demonstrating that
  conventional ``non-stationarity'' of data does not imply a non-zero
  inhomogeneity parameter of the dataset. We prove that - within the
  probabilistic Gaussian Process-based learning approach - a training
  set with a non-zero inhomogeneity parameter renders it imperative,
  that the process that is invoked to model the sought function, be
  non-stationary. Following the learning of a real-world multivariate
  function with such a Process, quality and reliability of predictions
  at test inputs, are demonstrated to be affected by the inhomogeneity
  parameter of the training data.

### 4. [Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](http://arxiv.org/pdf/2510.18638v1)

Authors: Yanna Ding, Songtao Lu, Yingdong Lu, Tomasz Nowicki, Jianxi Gao

Transformer architectures can solve unseen tasks based on input-output pairs
in a given prompt due to in-context learning (ICL). Existing theoretical
studies on ICL have mainly focused on linear regression tasks, often with
i.i.d. inputs. To understand how transformers express ICL when modeling
dynamics-driven functions, we investigate Markovian function learning through a
structured ICL setup, where we characterize the loss landscape to reveal
underlying optimization behaviors. Specifically, we (1) provide the closed-form
expression of the global minimizer (in an enlarged parameter space) for a
single-layer linear self-attention (LSA) model; (2) prove that recovering
transformer parameters that realize the optimal solution is NP-hard in general,
revealing a fundamental limitation of one-layer LSA in representing structured
dynamical functions; and (3) supply a novel interpretation of a multilayer LSA
as performing preconditioned gradient descent to optimize multiple objectives
beyond the square loss. These theoretical results are numerically validated
using simplified transformers.

### 5. [FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo](http://arxiv.org/pdf/2510.18193v1)

Authors: Keivan Shariatmadar, Ahmad Osman, Ramin Ray, Usman Dildar, Kisam Kim

Fair, transparent, and explainable decision-making remains a critical
challenge in Olympic and Paralympic combat sports. This paper presents
\emph{FST.ai 2.0}, an explainable AI ecosystem designed to support referees,
coaches, and athletes in real time during Taekwondo competitions and training.
The system integrates {pose-based action recognition} using graph convolutional
networks (GCNs), {epistemic uncertainty modeling} through credal sets, and
{explainability overlays} for visual decision support. A set of {interactive
dashboards} enables human--AI collaboration in referee evaluation, athlete
performance analysis, and Para-Taekwondo classification. Beyond automated
scoring, FST.ai~2.0 incorporates modules for referee training, fairness
monitoring, and policy-level analytics within the World Taekwondo ecosystem.
Experimental validation on competition data demonstrates an {85\% reduction in
decision review time} and {93\% referee trust} in AI-assisted decisions. The
framework thus establishes a transparent and extensible pipeline for
trustworthy, data-driven officiating and athlete assessment. By bridging
real-time perception, explainable inference, and governance-aware design,
FST.ai~2.0 represents a step toward equitable, accountable, and human-aligned
AI in sports.

### 6. [The Picard-Lagrange Framework for Higher-Order Langevin Monte Carlo](http://arxiv.org/pdf/2510.18242v1)

Authors: Jaideep Mahajan, Kaihong Zhang, Feng Liang, Jingbo Liu

Sampling from log-concave distributions is a central problem in statistics
and machine learning. Prior work establishes theoretical guarantees for
Langevin Monte Carlo algorithm based on overdamped and underdamped Langevin
dynamics and, more recently, some third-order variants. In this paper, we
introduce a new sampling algorithm built on a general $K$th-order Langevin
dynamics, extending beyond second- and third-order methods. To discretize the
$K$th-order dynamics, we approximate the drift induced by the potential via
Lagrange interpolation and refine the node values at the interpolation points
using Picard-iteration corrections, yielding a flexible scheme that fully
utilizes the acceleration of higher-order Langevin dynamics. For targets with
smooth, strongly log-concave densities, we prove dimension-dependent
convergence in Wasserstein distance: the sampler achieves
$\varepsilon$-accuracy within $\widetilde
O(d^{\frac{K-1}{2K-3}}\varepsilon^{-\frac{2}{2K-3}})$ gradient evaluations for
$K \ge 3$. To our best knowledge, this is the first sampling algorithm
achieving such query complexity. The rate improves with the order $K$
increases, yielding better rates than existing first to third-order approaches.

### 7. [Learning under Quantization for High-Dimensional Linear Regression](http://arxiv.org/pdf/2510.18259v1)

Authors: Dechen Zhang, Junwei Su, Difan Zou

The use of low-bit quantization has emerged as an indispensable technique for
enabling the efficient training of large-scale models. Despite its widespread
empirical success, a rigorous theoretical understanding of its impact on
learning performance remains notably absent, even in the simplest linear
regression setting. We present the first systematic theoretical study of this
fundamental question, analyzing finite-step stochastic gradient descent (SGD)
for high-dimensional linear regression under a comprehensive range of
quantization targets: data, labels, parameters, activations, and gradients. Our
novel analytical framework establishes precise algorithm-dependent and
data-dependent excess risk bounds that characterize how different quantization
affects learning: parameter, activation, and gradient quantization amplify
noise during training; data quantization distorts the data spectrum; and data
and label quantization introduce additional approximation and quantized error.
Crucially, we prove that for multiplicative quantization (with input-dependent
quantization step), this spectral distortion can be eliminated, and for
additive quantization (with constant quantization step), a beneficial scaling
effect with batch size emerges. Furthermore, for common polynomial-decay data
spectra, we quantitatively compare the risks of multiplicative and additive
quantization, drawing a parallel to the comparison between FP and integer
quantization methods. Our theory provides a powerful lens to characterize how
quantization shapes the learning dynamics of optimization algorithms, paving
the way to further explore learning theory under practical hardware
constraints.

### 8. [Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks](http://arxiv.org/pdf/2510.18435v1)

Authors: Brandon Livio Annesi, Dario Bocchi, Chiara Cammarota

High-dimensional non-convex loss landscapes play a central role in the theory
of Machine Learning. Gaining insight into how these landscapes interact with
gradient-based optimization methods, even in relatively simple models, can shed
light on this enigmatic feature of neural networks. In this work, we will focus
on a prototypical simple learning problem, which generalizes the Phase
Retrieval inference problem by allowing the exploration of overparametrized
settings. Using techniques from field theory, we analyze the spectrum of the
Hessian at initialization and identify a Baik-Ben Arous-P\'ech\'e (BBP)
transition in the amount of data that separates regimes where the
initialization is informative or uninformative about a planted signal of a
teacher-student setup. Crucially, we demonstrate how overparameterization can
bend the loss landscape, shifting the transition point, even reaching the
information-theoretic weak-recovery threshold in the large overparameterization
limit, while also altering its qualitative nature. We distinguish between
continuous and discontinuous BBP transitions and support our analytical
predictions with simulations, examining how they compare to the finite-N
behavior. In the case of discontinuous BBP transitions strong finite-N
corrections allow the retrieval of information at a signal-to-noise ratio (SNR)
smaller than the predicted BBP transition. In these cases we provide estimates
for a new lower SNR threshold that marks the point at which initialization
becomes entirely uninformative.

### 9. [Interval Prediction of Annual Average Daily Traffic on Local Roads via Quantile Random Forest with High-Dimensional Spatial Data](http://arxiv.org/pdf/2510.18548v1)

Authors: Ying Yao, Daniel J. Graham

Accurate annual average daily traffic (AADT) data are vital for transport
planning and infrastructure management. However, automatic traffic detectors
across national road networks often provide incomplete coverage, leading to
underrepresentation of minor roads. While recent machine learning advances have
improved AADT estimation at unmeasured locations, most models produce only
point predictions and overlook estimation uncertainty. This study addresses
that gap by introducing an interval prediction approach that explicitly
quantifies predictive uncertainty. We integrate a Quantile Random Forest model
with Principal Component Analysis to generate AADT prediction intervals,
providing plausible traffic ranges bounded by estimated minima and maxima.
Using data from over 2,000 minor roads in England and Wales, and evaluated with
specialized interval metrics, the proposed method achieves an interval coverage
probability of 88.22%, a normalized average width of 0.23, and a Winkler Score
of 7,468.47. By combining machine learning with spatial and high-dimensional
analysis, this framework enhances both the accuracy and interpretability of
AADT estimation, supporting more robust and informed transport planning.

### 10. [Differentially Private E-Values](http://arxiv.org/pdf/2510.18654v1)

Authors: Daniel Csillag, Diego Mesquita

E-values have gained prominence as flexible tools for statistical inference
and risk control, enabling anytime- and post-hoc-valid procedures under minimal
assumptions. However, many real-world applications fundamentally rely on
sensitive data, which can be leaked through e-values. To ensure their safe
release, we propose a general framework to transform non-private e-values into
differentially private ones. Towards this end, we develop a novel biased
multiplicative noise mechanism that ensures our e-values remain statistically
valid. We show that our differentially private e-values attain strong
statistical power, and are asymptotically as powerful as their non-private
counterparts. Experiments across online risk monitoring, private healthcare,
and conformal e-prediction demonstrate our approach's effectiveness and
illustrate its broad applicability.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-22 PST.

### 1. [Discovering state-of-the-art reinforcement learning algorithms](https://www.nature.com/articles/s41586-025-09761-x)

Authors: Junhyuk Oh et al.

### 2. [Google AI aims to make best-in-class scientific software even better](https://www.nature.com/articles/d41586-025-03289-w)

Authors: Matthew Hutson

### 3. [AI discovers learning algorithm that outperforms those designed by humans](https://www.nature.com/articles/d41586-025-03398-6)

Authors: Joel Lehman

### 4. [Optimization by decoded quantum interferometry](https://www.nature.com/articles/s41586-025-09527-5)

Authors: Stephen P. Jordan et al.

### 5. [A new hybrid neural network framework inspired by biological systems for advanced financial forecasting](https://www.nature.com/articles/s41598-025-21842-5)

Authors: Chuchu Rao et al.

### 6. [A novel multi objective energy efficient clustering optimization scheme based on heuristic intelligence for wireless sensor networks](https://www.nature.com/articles/s41598-025-21918-2)

Authors: Chuchu Rao et al.

### 7. [Efficient progressive training with granularity cross for image super-resolution](https://www.nature.com/articles/s41598-025-20975-x)

Authors: Yanzhen Lin et al.

### 8. [Integrating artificial intelligence and sustainable materials for smart eco innovation in production](https://www.nature.com/articles/s41598-025-20803-2)

Authors: Xingsi Xue et al.

### 9. [Generative artificial intelligence models outperform students on divergent and convergent thinking assessments](https://www.nature.com/articles/s41598-025-21398-4)

Authors: Vikram Arora et al.

### 10. [Multi-objective optimization of electromagnetic vibration parameters for corn seed phenotype prediction based on deep learning](https://www.nature.com/articles/s41598-025-20846-5)

Authors: Xinwei Zhang et al.

### 11. [Efficient classical sampling from Gaussian boson sampling distributions on unweighted graphs](https://www.nature.com/articles/s41467-025-64442-7)

Authors: Yexin Zhang et al.

