# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-20 17:00:26.860912 PST.

### Artificial Intelligence

### 1. [STPFormer: A State-of-the-Art Pattern-Aware Spatio-Temporal Transformer for Traffic Forecasting](http://arxiv.org/pdf/2508.13433v1)

Authors: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao

Spatio-temporal traffic forecasting is challenging due to complex temporal
patterns, dynamic spatial structures, and diverse input formats. Although
Transformer-based models offer strong global modeling, they often struggle with
rigid temporal encoding and weak space-time fusion. We propose STPFormer, a
Spatio-Temporal Pattern-Aware Transformer that achieves state-of-the-art
performance via unified and interpretable representation learning. It
integrates four modules: Temporal Position Aggregator (TPA) for pattern-aware
temporal encoding, Spatial Sequence Aggregator (SSA) for sequential spatial
learning, Spatial-Temporal Graph Matching (STGM) for cross-domain alignment,
and an Attention Mixer for multi-scale fusion. Experiments on five real-world
datasets show that STPFormer consistently sets new SOTA results, with ablation
and visualizations confirming its effectiveness and generalizability.

### 2. [Discrete Optimization of Min-Max Violation and its Applications Across Computational Sciences](http://arxiv.org/pdf/2508.13437v1)

Authors: Cheikh Ahmed, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou

We introduce the Discrete Min-Max Violation (DMMV) as a general optimization
problem which seeks an assignment of discrete values to variables that
minimizes the largest constraint violation. This context-free mathematical
formulation is applicable to a wide range of use cases that have worst-case
performance requirements. After defining the DMMV problem mathematically, we
explore its properties to establish a foundational understanding. To tackle
DMMV instance sizes of practical relevance, we develop a GPU-accelerated
heuristic that takes advantage of the mathematical properties of DMMV for
speeding up the solution process. We demonstrate the versatile applicability of
our heuristic by solving three optimization problems as use cases: (1)
post-training quantization of language models, (2) discrete tomography, and (3)
Finite Impulse Response (FIR) filter design. In quantization without outlier
separation, our heuristic achieves 14% improvement on average over existing
methods. In discrete tomography, it reduces reconstruction error by 16% under
uniform noise and accelerates computations by a factor of 6 on GPU. For FIR
filter design, it nearly achieves 50% ripple reduction compared to using the
commercial integer optimization solver, Gurobi. Our comparative results point
to the benefits of studying DMMV as a context-free optimization problem and the
advantages that our proposed heuristic offers on three distinct problems. Our
GPU-accelerated heuristic will be made open-source to further stimulate
research on DMMV and its other applications. The code is available at
https://anonymous.4open.science/r/AMVM-5F3E/

### 3. [LM Agents May Fail to Act on Their Own Risk Knowledge](http://arxiv.org/pdf/2508.13465v1)

Authors: Yuzhi Tang, Tianxiao Li, Elizabeth Li, Chris J. Maddison, Honghua Dong, Yangjun Ruan

Language model (LM) agents have demonstrated significant potential for
automating real-world tasks, yet they pose a diverse array of potential, severe
risks in safety-critical scenarios. In this work, we identify a significant gap
between LM agents' risk awareness and safety execution abilities: while they
often answer "Yes" to queries like "Is executing `sudo rm -rf /*' dangerous?",
they will likely fail to identify such risks in instantiated trajectories or
even directly perform these risky actions when acting as agents. To
systematically investigate this, we develop a comprehensive evaluation
framework to examine agents' safety across three progressive dimensions: 1)
their knowledge about potential risks, 2) their ability to identify
corresponding risks in execution trajectories, and 3) their actual behaviors to
avoid executing these risky actions. Our evaluation reveals two critical
performance gaps that resemble the generator-validator gaps observed in LMs:
while agents demonstrate near-perfect risk knowledge ($>98\%$ pass rates), they
fail to apply this knowledge when identifying risks in actual scenarios (with
performance dropping by $>23\%$) and often still execute risky actions ($<26\%$
pass rates). Notably, this trend persists across more capable LMs as well as in
specialized reasoning models like DeepSeek-R1, indicating that simply scaling
model capabilities or inference compute does not inherently resolve safety
concerns. Instead, we take advantage of these observed gaps to develop a risk
verifier that independently critiques the proposed actions by agents, with an
abstractor that converts specific execution trajectories into abstract
descriptions where LMs can more effectively identify the risks. Our overall
system achieves a significant reduction of risky action execution by $55.3\%$
over vanilla-prompted agents.

### 4. [CrafterDojo: A Suite of Foundation Models for Building Open-Ended Embodied Agents in Crafter](http://arxiv.org/pdf/2508.13530v1)

Authors: Junyeong Park, Hyeonseo Cho, Sungjin Ahn

Developing general-purpose embodied agents is a core challenge in AI.
Minecraft provides rich complexity and internet-scale data, but its slow speed
and engineering overhead make it unsuitable for rapid prototyping. Crafter
offers a lightweight alternative that retains key challenges from Minecraft,
yet its use has remained limited to narrow tasks due to the absence of
foundation models that have driven progress in the Minecraft setting. In this
paper, we present CrafterDojo, a suite of foundation models and tools that
unlock the Crafter environment as a lightweight, prototyping-friendly, and
Minecraft-like testbed for general-purpose embodied agent research. CrafterDojo
addresses this by introducing CrafterVPT, CrafterCLIP, and CrafterSteve-1 for
behavior priors, vision-language grounding, and instruction following,
respectively. In addition, we provide toolkits for generating behavior and
caption datasets (CrafterPlay and CrafterCaption), reference agent
implementations, benchmark evaluations, and a complete open-source codebase.

### 5. [Toward Better EHR Reasoning in LLMs: Reinforcement Learning with Expert Attention Guidance](http://arxiv.org/pdf/2508.13579v1)

Authors: Yue Fang, Yuxin Guo, Jiaran Gao, Hongxin Ding, Xinke Jiang, Weibin Liao, Yongxin Xu, Yinghao Zhu, Zhibang Yang, Liantao Ma, Junfeng Zhao, Yasha Wang

Improving large language models (LLMs) for electronic health record (EHR)
reasoning is essential for enabling accurate and generalizable clinical
predictions. While LLMs excel at medical text understanding, they underperform
on EHR-based prediction tasks due to challenges in modeling temporally
structured, high-dimensional data. Existing approaches often rely on hybrid
paradigms, where LLMs serve merely as frozen prior retrievers while downstream
deep learning (DL) models handle prediction, failing to improve the LLM's
intrinsic reasoning capacity and inheriting the generalization limitations of
DL models. To this end, we propose EAG-RL, a novel two-stage training framework
designed to intrinsically enhance LLMs' EHR reasoning ability through expert
attention guidance, where expert EHR models refer to task-specific DL models
trained on EHR data. Concretely, EAG-RL first constructs high-quality, stepwise
reasoning trajectories using expert-guided Monte Carlo Tree Search to
effectively initialize the LLM's policy. Then, EAG-RL further optimizes the
policy via reinforcement learning by aligning the LLM's attention with
clinically salient features identified by expert EHR models. Extensive
experiments on two real-world EHR datasets show that EAG-RL improves the
intrinsic EHR reasoning ability of LLMs by an average of 14.62%, while also
enhancing robustness to feature perturbations and generalization to unseen
clinical domains. These results demonstrate the practical potential of EAG-RL
for real-world deployment in clinical prediction tasks. Our code have been
available at https://github.com/devilran6/EAG-RL.

### 6. [V2P: From Background Suppression to Center Peaking for Robust GUI Grounding Task](http://arxiv.org/pdf/2508.13634v1)

Authors: Jikai Chen, Long Chen, Dong Wang, Leilei Gan, Chenyi Zhuang, Jinjie Gu

Precise localization of GUI elements is crucial for the development of GUI
agents. Traditional methods rely on bounding box or center-point regression,
neglecting spatial interaction uncertainty and visual-semantic hierarchies.
Recent methods incorporate attention mechanisms but still face two key issues:
(1) ignoring processing background regions causes attention drift from the
desired area, and (2) uniform labeling fails to distinguish between center and
edges of the target UI element, leading to click imprecision. Inspired by how
humans visually process and interact with GUI elements, we propose the
Valley-to-Peak (V2P) method to address these issues. To mitigate background
distractions, V2P introduces a suppression attention mechanism that minimizes
the model's focus on irrelevant regions to highlight the intended region. For
the issue of center-edge distinction, V2P applies a Fitts' Law-inspired
approach by modeling GUI interactions as 2D Gaussian heatmaps where the weight
gradually decreases from the center towards the edges. The weight distribution
follows a Gaussian function, with the variance determined by the target's size.
Consequently, V2P effectively isolates the target area and teaches the model to
concentrate on the most essential point of the UI element. The model trained by
V2P achieves the performance with 92.3% and 50.5% on two benchmarks
ScreenSpot-v2 and ScreenSpot-Pro. Ablations further confirm each component's
contribution, highlighting V2P's generalizability for precise GUI grounding
tasks.

### 7. [ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings](http://arxiv.org/pdf/2508.13672v1)

Authors: Rehan Raza, Guanjin Wang, Kevin Wong, Hamid Laga, Marco Fisichella

Explainable Artificial Intelligence (XAI) methods, such as Local
Interpretable Model-Agnostic Explanations (LIME), have advanced the
interpretability of black-box machine learning models by approximating their
behavior locally using interpretable surrogate models. However, LIME's inherent
randomness in perturbation and sampling can lead to locality and instability
issues, especially in scenarios with limited training data. In such cases, data
scarcity can result in the generation of unrealistic variations and samples
that deviate from the true data manifold. Consequently, the surrogate model may
fail to accurately approximate the complex decision boundary of the original
model. To address these challenges, we propose a novel Instance-based Transfer
Learning LIME framework (ITL-LIME) that enhances explanation fidelity and
stability in data-constrained environments. ITL-LIME introduces instance
transfer learning into the LIME framework by leveraging relevant real instances
from a related source domain to aid the explanation process in the target
domain. Specifically, we employ clustering to partition the source domain into
clusters with representative prototypes. Instead of generating random
perturbations, our method retrieves pertinent real source instances from the
source cluster whose prototype is most similar to the target instance. These
are then combined with the target instance's neighboring real instances. To
define a compact locality, we further construct a contrastive learning-based
encoder as a weighting mechanism to assign weights to the instances from the
combined set based on their proximity to the target instance. Finally, these
weighted source and target instances are used to train the surrogate model for
explanation purposes.

### 8. [Knowledge Graph Completion for Action Prediction on Situational Graphs -- A Case Study on Household Tasks](http://arxiv.org/pdf/2508.13675v1)

Authors: Mariam Arustashvili, Jörg Deigmöller, Heiko Paulheim

Knowledge Graphs are used for various purposes, including business
applications, biomedical analyses, or digital twins in industry 4.0. In this
paper, we investigate knowledge graphs describing household actions, which are
beneficial for controlling household robots and analyzing video footage. In the
latter case, the information extracted from videos is notoriously incomplete,
and completing the knowledge graph for enhancing the situational picture is
essential. In this paper, we show that, while a standard link prediction
problem, situational knowledge graphs have special characteristics that render
many link prediction algorithms not fit for the job, and unable to outperform
even simple baselines.

### 9. [MHSNet:An MoE-based Hierarchical Semantic Representation Network for Accurate Duplicate Resume Detection with Large Language Model](http://arxiv.org/pdf/2508.13676v1)

Authors: Yu Li, Zulong Chen, Wenjian Xu, Hong Wen, Yipeng Yu, Man Lung Yiu, Yuyu Yin

To maintain the company's talent pool, recruiters need to continuously search
for resumes from third-party websites (e.g., LinkedIn, Indeed). However,
fetched resumes are often incomplete and inaccurate. To improve the quality of
third-party resumes and enrich the company's talent pool, it is essential to
conduct duplication detection between the fetched resumes and those already in
the company's talent pool. Such duplication detection is challenging due to the
semantic complexity, structural heterogeneity, and information incompleteness
of resume texts. To this end, we propose MHSNet, an multi-level identity
verification framework that fine-tunes BGE-M3 using contrastive learning. With
the fine-tuned , Mixture-of-Experts (MoE) generates multi-level sparse and
dense representations for resumes, enabling the computation of corresponding
multi-level semantic similarities. Moreover, the state-aware Mixture-of-Experts
(MoE) is employed in MHSNet to handle diverse incomplete resumes. Experimental
results verify the effectiveness of MHSNet

### 10. [The DeepLog Neurosymbolic Machine](http://arxiv.org/pdf/2508.13697v1)

Authors: Vincent Derkinderen, Robin Manhaeve, Rik Adriaensen, Lucas Van Praet, Lennert De Smet, Giuseppe Marra, Luc De Raedt

We contribute a theoretical and operational framework for neurosymbolic AI
called DeepLog. DeepLog introduces building blocks and primitives for
neurosymbolic AI that make abstraction of commonly used representations and
computational mechanisms used in neurosymbolic AI. DeepLog can represent and
emulate a wide range of neurosymbolic systems. It consists of two key
components. The first is the DeepLog language for specifying neurosymbolic
models and inference tasks. This language consists of an annotated neural
extension of grounded first-order logic, and makes abstraction of the type of
logic, e.g. boolean, fuzzy or probabilistic, and whether logic is used in the
architecture or in the loss function. The second DeepLog component is situated
at the computational level and uses extended algebraic circuits as
computational graphs. Together these two components are to be considered as a
neurosymbolic abstract machine, with the DeepLog language as the intermediate
level of abstraction and the circuits level as the computational one. DeepLog
is implemented in software, relies on the latest insights in implementing
algebraic circuits on GPUs, and is declarative in that it is easy to obtain
different neurosymbolic models by making different choices for the underlying
algebraic structures and logics. The generality and efficiency of the DeepLog
neurosymbolic machine is demonstrated through an experimental comparison
between 1) different fuzzy and probabilistic logics, 2) between using logic in
the architecture or in the loss function, and 3) between a standalone CPU-based
implementation of a neurosymbolic AI system and a DeepLog GPU-based one.

### Computational Complexity

### 1. [Adversarially robust quantum state learning and testing](http://arxiv.org/pdf/2508.13959v1)

Authors: Maryam Aliakbarpour, Vladimir Braverman, Nai-Hui Chia, Yuhan Liu

Quantum state learning is a fundamental problem in physics and computer
science. As near-term quantum devices are error-prone, it is important to
design error-resistant algorithms. Apart from device errors, other unexpected
factors could also affect the algorithm, such as careless human read-out error,
or even a malicious hacker deliberately altering the measurement results. Thus,
we want our algorithm to work even in the worst case when things go against our
favor.
  We consider the practical setting of single-copy measurements and propose the
$\gamma$-adversarial corruption model where an imaginary adversary can
arbitrarily change $\gamma$-fraction of the measurement outcomes. This is
stronger than the $\gamma$-bounded SPAM noise model, where the post-measurement
state changes by at most $\gamma$ in trace distance. Under our stronger model
of corruption, we design an algorithm using non-adaptive measurements that can
learn an unknown rank-$r$ state up to $\tilde{O}(\gamma\sqrt{r})$ in trace
distance, provided that the number of copies is sufficiently large. We further
prove an information-theoretic lower bound of $\Omega(\gamma\sqrt{r})$ for
non-adaptive measurements, demonstrating the optimality of our algorithm. Our
upper and lower bounds also hold for quantum state testing, where the goal is
to test whether an unknown state is equal to a given state or far from it.
  Our results are intriguingly optimistic and pessimistic at the same time. For
general states, the error is dimension-dependent and $\gamma\sqrt{d}$ in the
worst case, meaning that only corrupting a very small fraction ($1/\sqrt{d}$)
of the outcomes could totally destroy any non-adaptive learning algorithm.
However, for constant-rank states that are useful in many quantum algorithms,
it is possible to achieve dimension-independent error, even in the worst-case
adversarial setting.

### 2. [Analog computation with transcriptional networks](http://arxiv.org/pdf/2508.14017v1)

Authors: David Doty, Mina Latifi, David Soloveichick

Transcriptional networks represent one of the most extensively studied types
of systems in synthetic biology. Although the completeness of transcriptional
networks for digital logic is well-established, *analog* computation plays a
crucial role in biological systems and offers significant potential for
synthetic biology applications. While transcriptional circuits typically rely
on cooperativity and highly non-linear behavior of transcription factors to
regulate *production* of proteins, they are often modeled with simple linear
*degradation* terms. In contrast, general analog dynamics require both
non-linear positive as well as negative terms, seemingly necessitating control
over not just transcriptional (i.e., production) regulation but also the
degradation rates of transcription factors.
  Surprisingly, we prove that controlling transcription factor production
(i.e., transcription rate) without explicitly controlling degradation is
mathematically complete for analog computation, achieving equivalent
capabilities to systems where both production and degradation are programmable.
We demonstrate our approach on several examples including oscillatory and
chaotic dynamics, analog sorting, memory, PID controller, and analog extremum
seeking. Our result provides a systematic methodology for engineering novel
analog dynamics using synthetic transcriptional networks without the added
complexity of degradation control and informs our understanding of the
capabilities of natural transcriptional circuits.
  We provide a compiler, in the form of a Python package that can take any
system of polynomial ODEs and convert it to an equivalent transcriptional
network implementing the system *exactly*, under appropriate conditions.

### Computational Engineering

### 1. [From Scores to Skills: A Cognitive Diagnosis Framework for Evaluating Financial Large Language Models](http://arxiv.org/pdf/2508.13491v1)

Authors: Ziyan Kuang, Feiyu Zhu, Maowei Jiang, Yanzhao Lai, Zelin Wang, Zhitong Wang, Meikang Qiu, Jiajia Huang, Min Peng, Qianqian Xie, Sophia Ananiadou

Large Language Models (LLMs) have shown promise for financial applications,
yet their suitability for this high-stakes domain remains largely unproven due
to inadequacies in existing benchmarks. Existing benchmarks solely rely on
score-level evaluation, summarizing performance with a single score that
obscures the nuanced understanding of what models truly know and their precise
limitations. They also rely on datasets that cover only a narrow subset of
financial concepts, while overlooking other essentials for real-world
applications. To address these gaps, we introduce FinCDM, the first cognitive
diagnosis evaluation framework tailored for financial LLMs, enabling the
evaluation of LLMs at the knowledge-skill level, identifying what financial
skills and knowledge they have or lack based on their response patterns across
skill-tagged tasks, rather than a single aggregated number. We construct
CPA-QKA, the first cognitively informed financial evaluation dataset derived
from the Certified Public Accountant (CPA) examination, with comprehensive
coverage of real-world accounting and financial skills. It is rigorously
annotated by domain experts, who author, validate, and annotate questions with
high inter-annotator agreement and fine-grained knowledge labels. Our extensive
experiments on 30 proprietary, open-source, and domain-specific LLMs show that
FinCDM reveals hidden knowledge gaps, identifies under-tested areas such as tax
and regulatory reasoning overlooked by traditional benchmarks, and uncovers
behavioral clusters among models. FinCDM introduces a new paradigm for
financial LLM evaluation by enabling interpretable, skill-aware diagnosis that
supports more trustworthy and targeted model development, and all datasets and
evaluation scripts will be publicly released to support further research.

### 2. [Data-Driven Discovery of Multi-Dimensional Breakage Population Balance Equations](http://arxiv.org/pdf/2508.13763v1)

Authors: Suet Lin Leong, Firnaaz Ahamed, Yong Kuen Ho

Multi-dimensional breakage is a ubiquitous phenomenon in natural systems, yet
the systematic discovery of underlying governing equations remains a
long-standing challenge. Current inverse solution techniques are restricted to
one-dimensional cases and typically depend on the availability of a priori
system knowledge, thus limiting their applicability. By leveraging advances in
data-driven sparse regression techniques, we develop the Multi-Dimensional
Breakage Population Balance Equation Identification (mPBE ID) algorithm for
discovering multi-dimensional breakage population balance equations (mPBEs)
directly from data. Our mPBE-ID enables tractable identification of mPBEs by
incorporating several key strategies, namely, a breakage-informed constrained
sparse regression, targeted candidate library functions construction via
insights from Dynamic Mode Decomposition (DMD), and robust handling of
noisy/limited data through ensembling (bagging/bragging). Notably, we
demonstrate how the DMD is indispensable for distilling dominant breakage
dynamics which can then be used to facilitate the systematic inclusion of
candidate library terms. We showcase the ability of the mPBE-ID to discover
different forms of mPBE (including those with discontinuous stoichiometric
kernels) even when tested against noisy and limited data. We anticipate that
the mPBE-ID will serve as a foundational framework for future extensions to
generalize the discovery of multi-dimensional PBEs for various high-dimensional
particulate phenomena.

### Computation and Language

### 1. [MATA (māta): Mindful Assessment of the Telugu Abilities of Large Language Models](http://arxiv.org/pdf/2508.13526v1)

Authors: Chalamalasetti Kranti, Sowmya Vajjala

In this paper, we introduce MATA, a novel evaluation dataset to assess the
ability of Large Language Models (LLMs) in Telugu language, comprising 729
carefully curated multiple-choice and open-ended questions that span diverse
linguistic dimensions. We evaluate 11 open-weight and closed-source LLMs on our
dataset and present a fine-grained analysis of their performance. Further, we
empirically show how LLMs rely on superficial heuristics such as answer
position and distractor patterns for multiple-choice questions. Finally, we
also compare LLM-as-a-judge evaluation with human evaluation for open-ended
questions and draw some conclusions on its reliability in a low-resource
language. We argue that such fine-grained evaluation is essential for
understanding model limitations and can inform the development of more
linguistically capable LLMs, while also serving as a foundation for future
research in Telugu NLP.

### 2. [AdaDocVQA: Adaptive Framework for Long Document Visual Question Answering in Low-Resource Settings](http://arxiv.org/pdf/2508.13606v1)

Authors: Haoxuan Li, Wei Song, Aofan Liu, Peiwu Qin

Document Visual Question Answering (Document VQA) faces significant
challenges when processing long documents in low-resource environments due to
context limitations and insufficient training data. This paper presents
AdaDocVQA, a unified adaptive framework addressing these challenges through
three core innovations: a hybrid text retrieval architecture for effective
document segmentation, an intelligent data augmentation pipeline that
automatically generates high-quality reasoning question-answer pairs with
multi-level verification, and adaptive ensemble inference with dynamic
configuration generation and early stopping mechanisms. Experiments on Japanese
document VQA benchmarks demonstrate substantial improvements with 83.04\%
accuracy on Yes/No questions, 52.66\% on factual questions, and 44.12\% on
numerical questions in JDocQA, and 59\% accuracy on LAVA dataset. Ablation
studies confirm meaningful contributions from each component, and our framework
establishes new state-of-the-art results for Japanese document VQA while
providing a scalable foundation for other low-resource languages and
specialized domains. Our code available at:
https://github.com/Haoxuanli-Thu/AdaDocVQA.

### 3. [CRISP: Persistent Concept Unlearning via Sparse Autoencoders](http://arxiv.org/pdf/2508.13650v1)

Authors: Tomer Ashuach, Dana Arad, Aaron Mueller, Martin Tutek, Yonatan Belinkov

As large language models (LLMs) are increasingly deployed in real-world
applications, the need to selectively remove unwanted knowledge while
preserving model utility has become paramount. Recent work has explored sparse
autoencoders (SAEs) to perform precise interventions on monosemantic features.
However, most SAE-based methods operate at inference time, which does not
create persistent changes in the model's parameters. Such interventions can be
bypassed or reversed by malicious actors with parameter access. We introduce
CRISP, a parameter-efficient method for persistent concept unlearning using
SAEs. CRISP automatically identifies salient SAE features across multiple
layers and suppresses their activations. We experiment with two LLMs and show
that our method outperforms prior approaches on safety-critical unlearning
tasks from the WMDP benchmark, successfully removing harmful knowledge while
preserving general and in-domain capabilities. Feature-level analysis reveals
that CRISP achieves semantically coherent separation between target and benign
concepts, allowing precise suppression of the target features.

### 4. [EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation](http://arxiv.org/pdf/2508.13735v1)

Authors: Yi Wang, Haoran Luo, Lu Meng

With the widespread application of electroencephalography (EEG) in
neuroscience and clinical practice, efficiently retrieving and semantically
interpreting large-scale, multi-source, heterogeneous EEG data has become a
pressing challenge. We propose EEG-MedRAG, a three-layer hypergraph-based
retrieval-augmented generation framework that unifies EEG domain knowledge,
individual patient cases, and a large-scale repository into a traversable n-ary
relational hypergraph, enabling joint semantic-temporal retrieval and
causal-chain diagnostic generation. Concurrently, we introduce the first
cross-disease, cross-role EEG clinical QA benchmark, spanning seven disorders
and five authentic clinical perspectives. This benchmark allows systematic
evaluation of disease-agnostic generalization and role-aware contextual
understanding. Experiments show that EEG-MedRAG significantly outperforms
TimeRAG and HyperGraphRAG in answer accuracy and retrieval, highlighting its
strong potential for real-world clinical decision support. Our data and code
are publicly available at https://github.com/yi9206413-boop/EEG-MedRAG.

### 5. [Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA](http://arxiv.org/pdf/2508.13743v1)

Authors: Kaiwei Zhang, Qi Jia, Zijian Chen, Wei Sun, Xiangyang Zhu, Chunyi Li, Dandan Zhu, Guangtao Zhai

Large language models (LLMs), while increasingly used in domains requiring
factual rigor, often display a troubling behavior: sycophancy, the tendency to
align with user beliefs regardless of correctness. This tendency is reinforced
by preference-based alignment techniques that optimize for user satisfaction
but can undermine truthfulness. While relatively benign in casual dialogue,
sycophancy poses serious risks in high-stakes settings such as scientific
question answering (QA), where model outputs may shape collaborative reasoning,
decision-making, and knowledge formation. Despite its importance, this
phenomenon remains underexamined in factual QA contexts. We address this gap by
introducing a unified evaluation framework to quantify the impact of
sycophantic context on model behavior in scientific QA, measuring how much
user-imposed social pressure distorts model outputs. The framework incorporates
adversarial prompting setups and targeted metrics, such as misleading
resistance and sycophancy resistance, that capture a model's ability to
maintain factual consistency under misleading cues. Systematic evaluations
across open-source and proprietary models reveal pervasive sycophantic
tendencies, driven more by alignment strategy than by model size. To mitigate
this issue, we propose Pressure-Tune, a lightweight post-training method that
fine-tunes models on synthetic adversarial dialogues paired with
chain-of-thought rationales. These rationales reject user misinformation while
reinforcing factual commitments. Experiments on challenging scientific QA
benchmarks show that Pressure-Tune significantly enhances sycophancy resistance
without compromising accuracy or responsiveness to valid feedback, offering a
practical pathway toward more truthful and principled model behavior.

### 6. [MGT-Prism: Enhancing Domain Generalization for Machine-Generated Text Detection via Spectral Alignment](http://arxiv.org/pdf/2508.13768v1)

Authors: Shengchao Liu, Xiaoming Liu, Chengzhengxu Li, Zhaohan Zhang, Guoxin Ma, Yu Lan, Shuai Xiao

Large Language Models have shown growing ability to generate fluent and
coherent texts that are highly similar to the writing style of humans. Current
detectors for Machine-Generated Text (MGT) perform well when they are trained
and tested in the same domain but generalize poorly to unseen domains, due to
domain shift between data from different sources. In this work, we propose
MGT-Prism, an MGT detection method from the perspective of the frequency domain
for better domain generalization. Our key insight stems from analyzing text
representations in the frequency domain, where we observe consistent spectral
patterns across diverse domains, while significant discrepancies in magnitude
emerge between MGT and human-written texts (HWTs). The observation initiates
the design of a low frequency domain filtering module for filtering out the
document-level features that are sensitive to domain shift, and a dynamic
spectrum alignment strategy to extract the task-specific and domain-invariant
features for improving the detector's performance in domain generalization.
Extensive experiments demonstrate that MGT-Prism outperforms state-of-the-art
baselines by an average of 0.90% in accuracy and 0.92% in F1 score on 11 test
datasets across three domain-generalization scenarios.

### 7. [Can Large Language Models (LLMs) Describe Pictures Like Children? A Comparative Corpus Study](http://arxiv.org/pdf/2508.13769v1)

Authors: Hanna Woloszyn, Benjamin Gagl

The role of large language models (LLMs) in education is increasing, yet
little attention has been paid to whether LLM-generated text resembles child
language. This study evaluates how LLMs replicate child-like language by
comparing LLM-generated texts to a collection of German children's descriptions
of picture stories. We generated two LLM-based corpora using the same picture
stories and two prompt types: zero-shot and few-shot prompts specifying a
general age from the children corpus. We conducted a comparative analysis
across psycholinguistic text properties, including word frequency, lexical
richness, sentence and word length, part-of-speech tags, and semantic
similarity with word embeddings. The results show that LLM-generated texts are
longer but less lexically rich, rely more on high-frequency words, and
under-represent nouns. Semantic vector space analysis revealed low similarity,
highlighting differences between the two corpora on the level of corpus
semantics. Few-shot prompt increased similarities between children and LLM text
to a minor extent, but still failed to replicate lexical and semantic patterns.
The findings contribute to our understanding of how LLMs approximate child
language through multimodal prompting (text + image) and give insights into
their use in psycholinguistic research and education while raising important
questions about the appropriateness of LLM-generated language in child-directed
educational tools.

### 8. [TracSum: A New Benchmark for Aspect-Based Summarization with Sentence-Level Traceability in Medical Domain](http://arxiv.org/pdf/2508.13798v1)

Authors: Bohao Chu, Meijie Li, Sameh Frihat, Chengyu Gu, Georg Lodde, Elisabeth Livingstone, Norbert Fuhr

While document summarization with LLMs has enhanced access to textual
information, concerns about the factual accuracy of these summaries persist,
especially in the medical domain. Tracing evidence from which summaries are
derived enables users to assess their accuracy, thereby alleviating this
concern. In this paper, we introduce TracSum, a novel benchmark for traceable,
aspect-based summarization, in which generated summaries are paired with
sentence-level citations, enabling users to trace back to the original context.
First, we annotate 500 medical abstracts for seven key medical aspects,
yielding 3.5K summary-citation pairs. We then propose a fine-grained evaluation
framework for this new task, designed to assess the completeness and
consistency of generated content using four metrics. Finally, we introduce a
summarization pipeline, Track-Then-Sum, which serves as a baseline method for
comparison. In experiments, we evaluate both this baseline and a set of LLMs on
TracSum, and conduct a human evaluation to assess the evaluation results. The
findings demonstrate that TracSum can serve as an effective benchmark for
traceable, aspect-based summarization tasks. We also observe that explicitly
performing sentence-level tracking prior to summarization enhances generation
accuracy, while incorporating the full context further improves completeness.

### 9. [ReviewGraph: A Knowledge Graph Embedding Based Framework for Review Rating Prediction with Sentiment Features](http://arxiv.org/pdf/2508.13953v1)

Authors: A. J. W. de Vink, Natalia Amat-Lefort, Lifeng Han

In the hospitality industry, understanding the factors that drive customer
review ratings is critical for improving guest satisfaction and business
performance. This work proposes ReviewGraph for Review Rating Prediction (RRP),
a novel framework that transforms textual customer reviews into knowledge
graphs by extracting (subject, predicate, object) triples and associating
sentiment scores. Using graph embeddings (Node2Vec) and sentiment features, the
framework predicts review rating scores through machine learning classifiers.
We compare ReviewGraph performance with traditional NLP baselines (such as Bag
of Words, TF-IDF, and Word2Vec) and large language models (LLMs), evaluating
them in the HotelRec dataset. In comparison to the state of the art literature,
our proposed model performs similar to their best performing model but with
lower computational cost (without ensemble).
  While ReviewGraph achieves comparable predictive performance to LLMs and
outperforms baselines on agreement-based metrics such as Cohen's Kappa, it
offers additional advantages in interpretability, visual exploration, and
potential integration into Retrieval-Augmented Generation (RAG) systems. This
work highlights the potential of graph-based representations for enhancing
review analytics and lays the groundwork for future research integrating
advanced graph neural networks and fine-tuned LLM-based extraction methods. We
will share ReviewGraph output and platform open-sourced on our GitHub page
https://github.com/aaronlifenghan/ReviewGraph

### 10. [Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR](http://arxiv.org/pdf/2508.14029v1)

Authors: Xiao Liang, Zhongzhi Li, Yeyun Gong, Yelong Shen, Ying Nian Wu, Zhijiang Guo, Weizhu Chen

Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as
a key paradigm for post-training Large Language Models (LLMs), particularly for
complex reasoning tasks. However, vanilla RLVR training has been shown to
improve Pass@1 performance at the expense of policy entropy, leading to reduced
generation diversity and limiting the Pass@k performance, which typically
represents the upper bound of LLM reasoning capability. In this paper, we
systematically analyze the policy's generation diversity from the perspective
of training problems and find that augmenting and updating training problems
helps mitigate entropy collapse during training. Based on these observations,
we propose an online Self-play with Variational problem Synthesis (SvS)
strategy for RLVR training, which uses the policy's correct solutions to
synthesize variational problems while ensuring their reference answers remain
identical to the originals. This self-improving strategy effectively maintains
policy entropy during training and substantially improves Pass@k compared with
standard RLVR, sustaining prolonged improvements and achieving absolute gains
of 18.3% and 22.8% in Pass@32 performance on the competition-level AIME24 and
AIME25 benchmarks. Experiments on 12 reasoning benchmarks across varying model
sizes from 3B to 32B consistently demonstrate the generalizability and
robustness of SvS.

### Cryptography and Security

### 1. [When Secure Aggregation Falls Short: Achieving Long-Term Privacy in Asynchronous Federated Learning for LEO Satellite Networks](http://arxiv.org/pdf/2508.13425v1)

Authors: Mohamed Elmahallawy, Tie Luo

Secure aggregation is a common technique in federated learning (FL) for
protecting data privacy from both curious internal entities (clients or server)
and external adversaries (eavesdroppers). However, in dynamic and
resource-constrained environments such as low Earth orbit (LEO) satellite
networks, traditional secure aggregation methods fall short in two aspects: (1)
they assume continuous client availability while LEO satellite visibility is
intermittent and irregular; (2) they consider privacy in each communication
round but have overlooked the possible privacy leakage through multiple rounds.
To address these limitations, we propose LTP-FLEO, an asynchronous FL framework
that preserves long-term privacy (LTP) for LEO satellite networks. LTP-FLEO
introduces (i) privacy-aware satellite partitioning, which groups satellites
based on their predictable visibility to the server and enforces joint
participation; (ii) model age balancing, which mitigates the adverse impact of
stale model updates; and (iii) fair global aggregation, which treats satellites
of different visibility durations in an equitable manner. Theoretical analysis
and empirical validation demonstrate that LTP-FLEO effectively safeguards both
model and data privacy across multi-round training, promotes fairness in line
with satellite contributions, accelerates global convergence, and achieves
competitive model accuracy.

### 2. [Beneath the Mask: Can Contribution Data Unveil Malicious Personas in Open-Source Projects?](http://arxiv.org/pdf/2508.13453v1)

Authors: Ruby Nealon

In February 2024, after building trust over two years with project
maintainers by making a significant volume of legitimate contributions, GitHub
user "JiaT75" self-merged a version of the XZ Utils project containing a highly
sophisticated, well-disguised backdoor targeting sshd processes running on
systems with the backdoored package installed. A month later, this package
began to be distributed with popular Linux distributions until a Microsoft
employee discovered the backdoor while investigating how a recent system
upgrade impacted the performance of SSH authentication. Despite its potential
global impact, no tooling exists for monitoring and identifying anomalous
behavior by personas contributing to other open-source projects. This paper
demonstrates how Open Source Intelligence (OSINT) data gathered from GitHub
contributions, analyzed using graph databases and graph theory, can efficiently
identify anomalous behaviors exhibited by the "JiaT75" persona across other
open-source projects.

### 3. [CAI Fluency: A Framework for Cybersecurity AI Fluency](http://arxiv.org/pdf/2508.13588v1)

Authors: Víctor Mayoral-Vilches, Jasmin Wachter, Cristóbal R. J. Veas Chavez, Cathrin Schachner, Luis Javier Navarrete-Lozano, María Sanz-Gómez

This work introduces CAI Fluency, an an educational platform of the
Cybersecurity AI (CAI) framework dedicated to democratizing the knowledge and
application of cybersecurity AI tools in the global security community. The
main objective of the CAI framework is to accelerate the widespread adoption
and effective use of artificial intelligence-based cybersecurity solutions,
pathing the way to vibe-hacking, the cybersecurity analogon to vibe-coding.
  CAI Fluency builds upon the Framework for AI Fluency, adapting its three
modalities of human-AI interaction and four core competencies specifically for
cybersecurity applications. This theoretical foundation ensures that
practitioners develop not just technical skills, but also the critical thinking
and ethical awareness necessary for responsible AI use in security contexts.
  This technical report serves as a white-paper, as well as detailed
educational and practical guide that helps users understand the principles
behind the CAI framework, and educates them how to apply this knowledge in
their projects and real-world security contexts.

### 4. [NodeShield: Runtime Enforcement of Security-Enhanced SBOMs for Node.js](http://arxiv.org/pdf/2508.13750v1)

Authors: Eric Cornelissen, Musard Balliu

The software supply chain is an increasingly common attack vector for
malicious actors. The Node.js ecosystem has been subject to a wide array of
attacks, likely due to its size and prevalence. To counter such attacks, the
research community and practitioners have proposed a range of static and
dynamic mechanisms, including process- and language-level sandboxing,
permission systems, and taint tracking. Drawing on valuable insight from these
works, this paper studies a runtime protection mechanism for (the supply chain
of) Node.js applications with the ambitious goals of compatibility, automation,
minimal overhead, and policy conciseness.
  Specifically, we design, implement and evaluate NodeShield, a protection
mechanism for Node.js that enforces an application's dependency hierarchy and
controls access to system resources at runtime. We leverage the up-and-coming
SBOM standard as the source of truth for the dependency hierarchy of the
application, thus preventing components from stealthily abusing undeclared
components. We propose to enhance the SBOM with a notion of capabilities that
represents a set of related system resources a component may access. Our
proposed SBOM extension, the Capability Bill of Materials or CBOM, records the
required capabilities of each component, providing valuable insight into the
potential privileged behavior. NodeShield enforces the SBOM and CBOM at runtime
via code outlining (as opposed to inlining) with no modifications to the
original code or Node.js runtime, thus preventing unexpected, potentially
malicious behavior. Our evaluation shows that NodeShield can prevent over 98%
out of 67 known supply chain attacks while incurring minimal overhead on
servers at less than 1ms per request. We achieve this while maintaining broad
compatibility with vanilla Node.js and a concise policy language that consists
of at most 7 entries per dependency.

### 5. [Red Teaming Methodology for Design Obfuscation](http://arxiv.org/pdf/2508.13965v1)

Authors: Yuntao Liu, Abir Akib, Zelin Lu, Qian Xu, Ankur Srivastava, Gang Qu, David Kehlet, Nij Dorairaj

The main goal of design obfuscation schemes is to protect sensitive design
details from untrusted parties in the VLSI supply chain, including but not
limited to off-shore foundries and untrusted end users. In this work, we
provide a systematic red teaming approach to evaluate the security of design
obfuscation approaches. Specifically, we propose security metrics and
evaluation methodology for the scenarios where the adversary does not have
access to a working chip. A case study on the RIPPER tool developed by the
University of Florida indicates that more information is leaked about the
structure of the original design than commonly considered.

### 6. [Conflicting Scores, Confusing Signals: An Empirical Study of Vulnerability Scoring Systems](http://arxiv.org/pdf/2508.13644v1)

Authors: Viktoria Koscinski, Mark Nelson, Ahmet Okutan, Robert Falso, Mehdi Mirakhorli

Accurately assessing software vulnerabilities is essential for effective
prioritization and remediation. While various scoring systems exist to support
this task, their differing goals, methodologies and outputs often lead to
inconsistent prioritization decisions. This work provides the first
large-scale, outcome-linked empirical comparison of four publicly available
vulnerability scoring systems: the Common Vulnerability Scoring System (CVSS),
the Stakeholder-Specific Vulnerability Categorization (SSVC), the Exploit
Prediction Scoring System (EPSS), and the Exploitability Index. We use a
dataset of 600 real-world vulnerabilities derived from four months of
Microsoft's Patch Tuesday disclosures to investigate the relationships between
these scores, evaluate how they support vulnerability management task, how
these scores categorize vulnerabilities across triage tiers, and assess their
ability to capture the real-world exploitation risk. Our findings reveal
significant disparities in how scoring systems rank the same vulnerabilities,
with implications for organizations relying on these metrics to make
data-driven, risk-based decisions. We provide insights into the alignment and
divergence of these systems, highlighting the need for more transparent and
consistent exploitability, risk, and severity assessments.

### 7. [Know Me by My Pulse: Toward Practical Continuous Authentication on Wearable Devices via Wrist-Worn PPG](http://arxiv.org/pdf/2508.13690v1)

Authors: Wei Shao, Zequan Liang, Ruoyu Zhang, Ruijie Fang, Ning Miao, Ehsan Kourkchi, Setareh Rafatirad, Houman Homayoun, Chongzhou Fang

Biometric authentication using physiological signals offers a promising path
toward secure and user-friendly access control in wearable devices. While
electrocardiogram (ECG) signals have shown high discriminability, their
intrusive sensing requirements and discontinuous acquisition limit
practicality. Photoplethysmography (PPG), on the other hand, enables
continuous, non-intrusive authentication with seamless integration into
wrist-worn wearable devices. However, most prior work relies on high-frequency
PPG (e.g., 75 - 500 Hz) and complex deep models, which incur significant energy
and computational overhead, impeding deployment in power-constrained real-world
systems. In this paper, we present the first real-world implementation and
evaluation of a continuous authentication system on a smartwatch, We-Be Band,
using low-frequency (25 Hz) multi-channel PPG signals. Our method employs a
Bi-LSTM with attention mechanism to extract identity-specific features from
short (4 s) windows of 4-channel PPG. Through extensive evaluations on both
public datasets (PTTPPG) and our We-Be Dataset (26 subjects), we demonstrate
strong classification performance with an average test accuracy of 88.11%,
macro F1-score of 0.88, False Acceptance Rate (FAR) of 0.48%, False Rejection
Rate (FRR) of 11.77%, and Equal Error Rate (EER) of 2.76%. Our 25 Hz system
reduces sensor power consumption by 53% compared to 512 Hz and 19% compared to
128 Hz setups without compromising performance. We find that sampling at 25 Hz
preserves authentication accuracy, whereas performance drops sharply at 20 Hz
while offering only trivial additional power savings, underscoring 25 Hz as the
practical lower bound. Additionally, we find that models trained exclusively on
resting data fail under motion, while activity-diverse training improves
robustness across physiological states.

### 8. [Optimizing Scalar Selection in Elliptic Curve Cryptography Using Differential Evolution for Enhanced Security](http://arxiv.org/pdf/2508.13520v1)

Authors: Takreem Haider

Elliptic Curve Cryptography (ECC) is a fundamental component of modern
public-key cryptosystems that enable efficient and secure digital signatures,
key exchanges, and encryption. Its core operation, scalar multiplication,
denoted as $k \cdot P$, where $P$ is a base point and $k$ is a private scalar,
relies heavily on the secrecy and unpredictability of $k$. Conventionally, $k$
is selected using user input or pseudorandom number generators. However, in
resource-constrained environments with weak entropy sources, these approaches
may yield low-entropy or biased scalars, increasing susceptibility to
side-channel and key recovery attacks. To mitigate these vulnerabilities, we
introduce an optimization-driven scalar generation method that explicitly
maximizes bit-level entropy. Our approach uses differential evolution (DE), a
population-based metaheuristic algorithm, to search for scalars whose binary
representations exhibit maximal entropy, defined by an even and statistically
uniform distribution of ones and zeros. This reformulation of scalar selection
as an entropy-optimization problem enhances resistance to entropy-based
cryptanalytic techniques and improves overall unpredictability. Experimental
results demonstrate that DE-optimized scalars achieve entropy significantly
higher than conventionally generated scalars. The proposed method can be
integrated into existing ECC-based protocols, offering a deterministic, tunable
alternative to traditional randomness, ideal for applications in blockchain,
secure messaging, IoT, and other resource-constrained environments.

### 9. [DDoS Attacks in Cloud Computing: Detection and Prevention](http://arxiv.org/pdf/2508.13522v1)

Authors: Zain Ahmad, Musab Ahmad, Bilal Ahmad

DDoS attacks are one of the most prevalent and harmful cybersecurity threats
faced by organizations and individuals today. In recent years, the complexity
and frequency of DDoS attacks have increased significantly, making it
challenging to detect and mitigate them effectively. The study analyzes various
types of DDoS attacks, including volumetric, protocol, and application layer
attacks, and discusses the characteristics, impact, and potential targets of
each type. It also examines the existing techniques used for DDoS attack
detection, such as packet filtering, intrusion detection systems, and machine
learning-based approaches, and their strengths and limitations. Moreover, the
study explores the prevention techniques employed to mitigate DDoS attacks,
such as firewalls, rate limiting , CPP and ELD mechanism. It evaluates the
effectiveness of each approach and its suitability for different types of
attacks and environments. In conclusion, this study provides a comprehensive
overview of the different types of DDoS attacks, their detection, and
prevention techniques. It aims to provide insights and guidelines for
organizations and individuals to enhance their cybersecurity posture and
protect against DDoS attacks.

### 10. [Optimizing Region of Interest Selection for Effective Embedding in Video Steganography Based on Genetic Algorithms](http://arxiv.org/pdf/2508.13710v1)

Authors: Nizheen A. Ali, Ramadhan J. Mstafa

With the widespread use of the internet, there is an increasing need to
ensure the security and privacy of transmitted data. This has led to an
intensified focus on the study of video steganography, which is a technique
that hides data within a video cover to avoid detection. The effectiveness of
any steganography method depends on its ability to embed data without altering
the original video quality while maintaining high efficiency. This paper
proposes a new method to video steganography, which involves utilizing a
Genetic Algorithm (GA) for identifying the Region of Interest (ROI) in the
cover video. The ROI is the area in the video that is the most suitable for
data embedding. The secret data is encrypted using the Advanced Encryption
Standard (AES), which is a widely accepted encryption standard, before being
embedded into the cover video, utilizing up to 10% of the cover video. This
process ensures the security and confidentiality of the embedded data. The
performance metrics for assessing the proposed method are the Peak Signal to
Noise Ratio (PSNR) and the encoding and decoding time. The results show that
the proposed method has a high embedding capacity and efficiency, with a PSNR
ranging between 64 and 75 dBs, which indicates that the embedded data is almost
indistinguishable from the original video. Additionally, the method can encode
and decode data quickly, making it efficient for real time applications.

### Computer Vision and Pattern Recognition

### 1. [EDTalk++: Full Disentanglement for Controllable Talking Head Synthesis](http://arxiv.org/pdf/2508.13442v1)

Authors: Shuai Tan, Bin Ji

Achieving disentangled control over multiple facial motions and accommodating
diverse input modalities greatly enhances the application and entertainment of
the talking head generation. This necessitates a deep exploration of the
decoupling space for facial features, ensuring that they a) operate
independently without mutual interference and b) can be preserved to share with
different modal inputs, both aspects often neglected in existing methods. To
address this gap, this paper proposes EDTalk++, a novel full disentanglement
framework for controllable talking head generation. Our framework enables
individual manipulation of mouth shape, head pose, eye movement, and emotional
expression, conditioned on video or audio inputs. Specifically, we employ four
lightweight modules to decompose the facial dynamics into four distinct latent
spaces representing mouth, pose, eye, and expression, respectively. Each space
is characterized by a set of learnable bases whose linear combinations define
specific motions. To ensure independence and accelerate training, we enforce
orthogonality among bases and devise an efficient training strategy to allocate
motion responsibilities to each space without relying on external knowledge.
The learned bases are then stored in corresponding banks, enabling shared
visual priors with audio input. Furthermore, considering the properties of each
space, we propose an Audio-to-Motion module for audio-driven talking head
synthesis. Experiments are conducted to demonstrate the effectiveness of
EDTalk++.

### 2. [Revisiting MLLM Token Technology through the Lens of Classical Visual Coding](http://arxiv.org/pdf/2508.13460v1)

Authors: Jinming Liu, Junyan Lin, Yuntao Wei, Kele Shao, Keda Tao, Jianguo Huang, Xudong Yang, Zhibo Chen, Huan Wang, Xin Jin

Classical visual coding and Multimodal Large Language Model (MLLM) token
technology share the core objective - maximizing information fidelity while
minimizing computational cost. Therefore, this paper reexamines MLLM token
technology, including tokenization, token compression, and token reasoning,
through the established principles of long-developed visual coding area. From
this perspective, we (1) establish a unified formulation bridging token
technology and visual coding, enabling a systematic, module-by-module
comparative analysis; (2) synthesize bidirectional insights, exploring how
visual coding principles can enhance MLLM token techniques' efficiency and
robustness, and conversely, how token technology paradigms can inform the
design of next-generation semantic visual codecs; (3) prospect for promising
future research directions and critical unsolved challenges. In summary, this
study presents the first comprehensive and structured technology comparison of
MLLM token and visual coding, paving the way for more efficient multimodal
models and more powerful visual codecs simultaneously.

### 3. [MINR: Efficient Implicit Neural Representations for Multi-Image Encoding](http://arxiv.org/pdf/2508.13471v1)

Authors: Wenyong Zhou, Taiqiang Wu, Zhengwu Liu, Yuxin Cheng, Chen Zhang, Ngai Wong

Implicit Neural Representations (INRs) aim to parameterize discrete signals
through implicit continuous functions. However, formulating each image with a
separate neural network~(typically, a Multi-Layer Perceptron (MLP)) leads to
computational and storage inefficiencies when encoding multi-images. To address
this issue, we propose MINR, sharing specific layers to encode multi-image
efficiently. We first compare the layer-wise weight distributions for several
trained INRs and find that corresponding intermediate layers follow highly
similar distribution patterns. Motivated by this, we share these intermediate
layers across multiple images while preserving the input and output layers as
input-specific. In addition, we design an extra novel projection layer for each
image to capture its unique features. Experimental results on image
reconstruction and super-resolution tasks demonstrate that MINR can save up to
60\% parameters while maintaining comparable performance. Particularly, MINR
scales effectively to handle 100 images, maintaining an average peak
signal-to-noise ratio (PSNR) of 34 dB. Further analysis of various backbones
proves the robustness of the proposed MINR.

### 4. [Distribution-Aware Hadamard Quantization for Hardware-Efficient Implicit Neural Representations](http://arxiv.org/pdf/2508.13478v1)

Authors: Wenyong Zhou, Jiachen Ren, Taiqiang Wu, Yuxin Cheng, Zhengwu Liu, Ngai Wong

Implicit Neural Representations (INRs) encode discrete signals using
Multi-Layer Perceptrons (MLPs) with complex activation functions. While INRs
achieve superior performance, they depend on full-precision number
representation for accurate computation, resulting in significant hardware
overhead. Previous INR quantization approaches have primarily focused on weight
quantization, offering only limited hardware savings due to the lack of
activation quantization. To fully exploit the hardware benefits of
quantization, we propose DHQ, a novel distribution-aware Hadamard quantization
scheme that targets both weights and activations in INRs. Our analysis shows
that the weights in the first and last layers have distributions distinct from
those in the intermediate layers, while the activations in the last layer
differ significantly from those in the preceding layers. Instead of customizing
quantizers individually, we utilize the Hadamard transformation to standardize
these diverse distributions into a unified bell-shaped form, supported by both
empirical evidence and theoretical analysis, before applying a standard
quantizer. To demonstrate the practical advantages of our approach, we present
an FPGA implementation of DHQ that highlights its hardware efficiency.
Experiments on diverse image reconstruction tasks show that DHQ outperforms
previous quantization methods, reducing latency by 32.7\%, energy consumption
by 40.1\%, and resource utilization by up to 98.3\% compared to full-precision
counterparts.

### 5. [Enhancing Robustness of Implicit Neural Representations Against Weight Perturbations](http://arxiv.org/pdf/2508.13481v1)

Authors: Wenyong Zhou, Yuxin Cheng, Zhengwu Liu, Taiqiang Wu, Chen Zhang, Ngai Wong

Implicit Neural Representations (INRs) encode discrete signals in a
continuous manner using neural networks, demonstrating significant value across
various multimedia applications. However, the vulnerability of INRs presents a
critical challenge for their real-world deployments, as the network weights
might be subjected to unavoidable perturbations. In this work, we investigate
the robustness of INRs for the first time and find that even minor
perturbations can lead to substantial performance degradation in the quality of
signal reconstruction. To mitigate this issue, we formulate the robustness
problem in INRs by minimizing the difference between loss with and without
weight perturbations. Furthermore, we derive a novel robust loss function to
regulate the gradient of the reconstruction loss with respect to weights,
thereby enhancing the robustness. Extensive experiments on reconstruction tasks
across multiple modalities demonstrate that our method achieves up to a 7.5~dB
improvement in peak signal-to-noise ratio (PSNR) values compared to original
INRs under noisy conditions.

### 6. [FAMNet: Integrating 2D and 3D Features for Micro-expression Recognition via Multi-task Learning and Hierarchical Attention](http://arxiv.org/pdf/2508.13483v1)

Authors: Liangyu Fu, Xuecheng Wu, Danlei Huang, Xinyi Yin

Micro-expressions recognition (MER) has essential application value in many
fields, but the short duration and low intensity of micro-expressions (MEs)
bring considerable challenges to MER. The current MER methods in deep learning
mainly include three data loading methods: static images, dynamic image
sequence, and a combination of the two streams. How to effectively extract MEs'
fine-grained and spatiotemporal features has been difficult to solve. This
paper proposes a new MER method based on multi-task learning and hierarchical
attention, which fully extracts MEs' omni-directional features by merging 2D
and 3D CNNs. The fusion model consists of a 2D CNN AMNet2D and a 3D CNN
AMNet3D, with similar structures consisting of a shared backbone network
Resnet18 and attention modules. During training, the model adopts different
data loading methods to adapt to two specific networks respectively, jointly
trains on the tasks of MER and facial action unit detection (FAUD), and adopts
the parameter hard sharing for information association, which further improves
the effect of the MER task, and the final fused model is called FAMNet.
Extensive experimental results show that our proposed FAMNet significantly
improves task performance. On the SAMM, CASME II and MMEW datasets, FAMNet
achieves 83.75% (UAR) and 84.03% (UF1). Furthermore, on the challenging
CAS(ME)$^3$ dataset, FAMNet achieves 51% (UAR) and 43.42% (UF1).

### 7. [Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models](http://arxiv.org/pdf/2508.13507v1)

Authors: Seungheon Baek, Jinhyuk Yun

Badminton is known as one of the fastest racket sports in the world. Despite
doubles matches being more prevalent in international tournaments than singles,
previous research has mainly focused on singles due to the challenges in data
availability and multi-person tracking. To address this gap, we designed an
approach that transfers singles-trained models to doubles analysis. We
extracted keypoints from the ShuttleSet single matches dataset using ViT-Pose
and embedded them through a contrastive learning framework based on ST-GCN. To
improve tracking stability, we incorporated a custom multi-object tracking
algorithm that resolves ID switching issues from fast and overlapping player
movements. A Transformer-based classifier then determines shot occurrences
based on the learned embeddings. Our findings demonstrate the feasibility of
extending pose-based shot recognition to doubles badminton, broadening
analytics capabilities. This work establishes a foundation for doubles-specific
datasets to enhance understanding of this predominant yet understudied format
of the fast racket sport.

### 8. [2D Gaussians Meet Visual Tokenizer](http://arxiv.org/pdf/2508.13515v1)

Authors: Yiang Shi, Xiaoyang Guo, Wei Yin, Mingkai Jia, Qian Zhang, Xiaolin Hu, Wenyu Liu, Xinggang Wan

The image tokenizer is a critical component in AR image generation, as it
determines how rich and structured visual content is encoded into compact
representations. Existing quantization-based tokenizers such as VQ-GAN
primarily focus on appearance features like texture and color, often neglecting
geometric structures due to their patch-based design. In this work, we explored
how to incorporate more visual information into the tokenizer and proposed a
new framework named Visual Gaussian Quantization (VGQ), a novel tokenizer
paradigm that explicitly enhances structural modeling by integrating 2D
Gaussians into traditional visual codebook quantization frameworks. Our
approach addresses the inherent limitations of naive quantization methods such
as VQ-GAN, which struggle to model structured visual information due to their
patch-based design and emphasis on texture and color. In contrast, VGQ encodes
image latents as 2D Gaussian distributions, effectively capturing geometric and
spatial structures by directly modeling structure-related parameters such as
position, rotation and scale. We further demonstrate that increasing the
density of 2D Gaussians within the tokens leads to significant gains in
reconstruction fidelity, providing a flexible trade-off between token
efficiency and visual richness. On the ImageNet 256x256 benchmark, VGQ achieves
strong reconstruction quality with an rFID score of 1.00. Furthermore, by
increasing the density of 2D Gaussians within the tokens, VGQ gains a
significant boost in reconstruction capability and achieves a state-of-the-art
reconstruction rFID score of 0.556 and a PSNR of 24.93, substantially
outperforming existing methods. Codes will be released soon.

### 9. [GazeProphet: Software-Only Gaze Prediction for VR Foveated Rendering](http://arxiv.org/pdf/2508.13546v1)

Authors: Farhaan Ebadulla, Chiraag Mudlapur, Gaurav BV

Foveated rendering significantly reduces computational demands in virtual
reality applications by concentrating rendering quality where users focus their
gaze. Current approaches require expensive hardware-based eye tracking systems,
limiting widespread adoption due to cost, calibration complexity, and hardware
compatibility constraints. This paper presents GazeProphet, a software-only
approach for predicting gaze locations in VR environments without requiring
dedicated eye tracking hardware. The approach combines a Spherical Vision
Transformer for processing 360-degree VR scenes with an LSTM-based temporal
encoder that captures gaze sequence patterns. A multi-modal fusion network
integrates spatial scene features with temporal gaze dynamics to predict future
gaze locations with associated confidence estimates. Experimental evaluation on
a comprehensive VR dataset demonstrates that GazeProphet achieves a median
angular error of 3.83 degrees, outperforming traditional saliency-based
baselines by 24% while providing reliable confidence calibration. The approach
maintains consistent performance across different spatial regions and scene
types, enabling practical deployment in VR systems without additional hardware
requirements. Statistical analysis confirms the significance of improvements
across all evaluation metrics. These results show that software-only gaze
prediction can work for VR foveated rendering, making this performance boost
more accessible to different VR platforms and apps.

### 10. [Color Spike Data Generation via Bio-inspired Neuron-like Encoding with an Artificial Photoreceptor Layer](http://arxiv.org/pdf/2508.13558v1)

Authors: Hsieh Ching-Teng, Wang Yuan-Kai

In recent years, neuromorphic computing and spiking neural networks (SNNs)
have ad-vanced rapidly through integration with deep learning. However, the
performance of SNNs still lags behind that of convolutional neural networks
(CNNs), primarily due to the limited information capacity of spike-based data.
Although some studies have attempted to improve SNN performance by training
them with non-spiking inputs such as static images, this approach deviates from
the original intent of neuromorphic computing, which emphasizes spike-based
information processing. To address this issue, we propose a Neuron-like
Encoding method that generates spike data based on the intrinsic operational
principles and functions of biological neurons. This method is further enhanced
by the incorporation of an artificial pho-toreceptor layer, enabling spike data
to carry both color and luminance information, thereby forming a complete
visual spike signal. Experimental results using the Integrate-and-Fire neuron
model demonstrate that this biologically inspired approach effectively
increases the information content of spike signals and improves SNN
performance, all while adhering to neuromorphic principles. We believe this
concept holds strong potential for future development and may contribute to
overcoming current limitations in neuro-morphic computing, facilitating broader
applications of SNNs.

### Computers and Society

### 1. [The AI-Fraud Diamond: A Novel Lens for Auditing Algorithmic Deception](http://arxiv.org/pdf/2508.13984v1)

Authors: Benjamin Zweers, Diptish Dey, Debarati Bhaumik

As artificial intelligence (AI) systems become increasingly integral to
organizational processes, they introduce new forms of fraud that are often
subtle, systemic, and concealed within technical complexity. This paper
introduces the AI-Fraud Diamond, an extension of the traditional Fraud Triangle
that adds technical opacity as a fourth condition alongside pressure,
opportunity, and rationalization. Unlike traditional fraud, AI-enabled
deception may not involve clear human intent but can arise from system-level
features such as opaque model behavior, flawed training data, or unregulated
deployment practices. The paper develops a taxonomy of AI-fraud across five
categories: input data manipulation, model exploitation, algorithmic decision
manipulation, synthetic misinformation, and ethics-based fraud. To assess the
relevance and applicability of the AI-Fraud Diamond, the study draws on expert
interviews with auditors from two of the Big Four consulting firms. The
findings underscore the challenges auditors face when addressing fraud in
opaque and automated environments, including limited technical expertise,
insufficient cross-disciplinary collaboration, and constrained access to
internal system processes. These conditions hinder fraud detection and reduce
accountability. The paper argues for a shift in audit methodology-from
outcome-based checks to a more diagnostic approach focused on identifying
systemic vulnerabilities. Ultimately, the work lays a foundation for future
empirical research and audit innovation in a rapidly evolving AI governance
landscape.

### 2. [Consumer Autonomy or Illusion? Rethinking Consumer Agency in the Age of Algorithms](http://arxiv.org/pdf/2508.13440v1)

Authors: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana

Consumer agency in the digital age is increasingly constrained by systemic
barriers and algorithmic manipulation, raising concerns about the authenticity
of consumption choices. Nowadays, financial decisions are shaped by external
pressures like obligatory consumption, algorithmic persuasion, and unstable
work schedules that erode financial autonomy. Obligatory consumption (like
hidden fees) is intensified by digital ecosystems. Algorithmic tactics like
personalized recommendations lead to impulsive purchases. Unstable work
schedules also undermine financial planning. Thus, it is important to study how
these factors impact consumption agency. To do so, we examine formal models
grounded in discounted consumption with constraints that bound agency. We
construct analytical scenarios in which consumers face obligatory payments,
algorithm-influenced impulsive expenses, or unpredictable income due to
temporal instability. Using this framework, we demonstrate that even rational,
utility-maximizing agents can experience early financial ruin when agency is
limited across structural, behavioral, or temporal dimensions and how
diminished autonomy impacts long-term financial well-being. Our central
argument is that consumer agency must be treated as a value (not a given)
requiring active cultivation, especially in digital ecosystems. The connection
between our formal modeling and this argument allows us to indicate that
limitations on agency (whether structural, behavioral, or temporal) can be
rigorously linked to measurable risks like financial instability. This
connection is also a basis for normative claims about consumption as a value,
by anchoring them in a formally grounded analysis of consumer behavior. As
solutions, we study systemic interventions and consumer education to support
value deliberation and informed choices. We formally demonstrate how these
measures strengthen agency.

### 3. [The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats](http://arxiv.org/pdf/2508.13700v1)

Authors: Markov Grey, Charbel-Raphaël Segerie

As AI systems become more capable, integrated, and widespread, understanding
the associated risks becomes increasingly important. This paper maps the full
spectrum of AI risks, from current harms affecting individual users to
existential threats that could endanger humanity's survival. We organize these
risks into three main causal categories. Misuse risks, which occur when people
deliberately use AI for harmful purposes - creating bioweapons, launching
cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons.
Misalignment risks happen when AI systems pursue outcomes that conflict with
human values, irrespective of developer intentions. This includes risks arising
through specification gaming (reward hacking), scheming and power-seeking
tendencies in pursuit of long-term strategic goals. Systemic risks, which arise
when AI integrates into complex social systems in ways that gradually undermine
human agency - concentrating power, accelerating political and economic
disempowerment, creating overdependence that leads to human enfeeblement, or
irreversibly locking in current values curtailing future moral progress. Beyond
these core categories, we identify risk amplifiers - competitive pressures,
accidents, corporate indifference, and coordination failures - that make all
risks more likely and severe. Throughout, we connect today's existing risks and
empirically observable AI behaviors to plausible future outcomes, demonstrating
how existing trends could escalate to catastrophic outcomes. Our goal is to
help readers understand the complete landscape of AI risks. Good futures are
possible, but they don't happen by default. Navigating these challenges will
require unprecedented coordination, but an extraordinary future awaits if we
do.

### 4. [Exit Stories: Using Reddit Self-Disclosures to Understand Disengagement from Problematic Communities](http://arxiv.org/pdf/2508.13837v1)

Authors: Shruti Phadke

Online platforms like Reddit are increasingly becoming popular for
individuals sharing personal experiences of leaving behind social, ideological,
and political groups. Specifically, a series of "ex-" subreddits on Reddit
allow users to recount their departures from commitments such as religious
affiliations, manosphere communities, conspiracy theories or political beliefs,
and lifestyle choices. Understanding the natural process through which users
exit, especially from problematic groups such as conspiracy theory communities
and the manosphere, can provide valuable insights for designing interventions
targeting disengagement from harmful ideologies. This paper presents an
in-depth exploration of 15K exit stories across 131 subreddits, focusing on
five key areas: religion, manosphere, conspiracy theories, politics, and
lifestyle. Using a transdisciplinary framework that incorporates theories from
social psychology, organizational behavior, and violent extremism studies, this
work identifies a range of factors contributing to disengagement. The results
describe how disengagement from problematic groups, such as conspiracy theories
and the manosphere, is a multi-faceted process that is qualitatively different
than disengaging from more established social structures, such as religions or
political ideologies. This research further highlights the need for moving
beyond interventions that treat conspiracy theorizing solely as an information
problem and contributes insights for future research focusing on offering
mental health interventions and support in exit communities.

### 5. [Trust and Reputation in Data Sharing: A Survey](http://arxiv.org/pdf/2508.14028v1)

Authors: Wenbo Wu, George Konstantinidis

Data sharing is the fuel of the galloping artificial intelligence economy,
providing diverse datasets for training robust models. Trust between data
providers and data consumers is widely considered one of the most important
factors for enabling data sharing initiatives. Concerns about data sensitivity,
privacy breaches, and misuse contribute to reluctance in sharing data across
various domains. In recent years, there has been a rise in technological and
algorithmic solutions to measure, capture and manage trust, trustworthiness,
and reputation in what we collectively refer to as Trust and Reputation
Management Systems (TRMSs). Such approaches have been developed and applied to
different domains of computer science, such as autonomous vehicles, or IoT
networks, but there have not been dedicated approaches to data sharing and its
unique characteristics. In this survey, we examine TRMSs from a data-sharing
perspective, analyzing how they assess the trustworthiness of both data and
entities across different environments. We develop novel taxonomies for system
designs, trust evaluation framework, and evaluation metrics for both data and
entity, and we systematically analyze the applicability of existing TRMSs in
data sharing. Finally, we identify open challenges and propose future research
directions to enhance the explainability, comprehensiveness, and accuracy of
TRMSs in large-scale data-sharing ecosystems.

### Databases

### 1. [Scavenger: Better Space-Time Trade-Offs for Key-Value Separated LSM-trees](http://arxiv.org/pdf/2508.13909v1)

Authors: Jianshun Zhang, Fang Wang, Sheng Qiu, Yi Wang, Jiaxin Ou, Junxun Huang, Baoquan Li, Peng Fang, Dan Feng

Key-Value Stores (KVS) implemented with log-structured merge-tree (LSM-tree)
have gained widespread acceptance in storage systems. Nonetheless, a
significant challenge arises in the form of high write amplification due to the
compaction process. While KV-separated LSM-trees successfully tackle this
issue, they also bring about substantial space amplification problems, a
concern that cannot be overlooked in cost-sensitive scenarios. Garbage
collection (GC) holds significant promise for space amplification reduction,
yet existing GC strategies often fall short in optimization performance,
lacking thorough consideration of workload characteristics. Additionally,
current KV-separated LSM-trees also ignore the adverse effect of the space
amplification in the index LSM-tree. In this paper, we systematically analyze
the sources of space amplification of KV-separated LSM-trees and introduce
Scavenger, which achieves a better trade-off between performance and space
amplification. Scavenger initially proposes an I/O-efficient garbage collection
scheme to reduce I/O overhead and incorporates a space-aware compaction
strategy based on compensated size to minimize the space amplification of index
LSM-trees. Extensive experiments show that Scavenger significantly improves
write performance and achieves lower space amplification than other
KV-separated LSM-trees (including BlobDB, Titan, and TerarkDB).

### 2. [Scavenger+: Revisiting Space-Time Tradeoffs in Key-Value Separated LSM-trees](http://arxiv.org/pdf/2508.13935v1)

Authors: Jianshun Zhang, Fang Wang, Jiaxin Ou, Yi Wang, Ming Zhao, Sheng Qiu, Junxun Huang, Baoquan Li, Peng Fang, Dan Feng

Key-Value Stores (KVS) based on log-structured merge-trees (LSM-trees) are
widely used in storage systems but face significant challenges, such as high
write amplification caused by compaction. KV-separated LSM-trees address write
amplification but introduce significant space amplification, a critical concern
in cost-sensitive scenarios. Garbage collection (GC) can reduce space
amplification, but existing strategies are often inefficient and fail to
account for workload characteristics. Moreover, current key-value (KV)
separated LSM-trees overlook the space amplification caused by the index
LSM-tree. In this paper, we systematically analyze the sources of space
amplification in KV-separated LSM-trees and propose Scavenger+, which achieves
a better performance-space trade-off. Scavenger+ introduces (1) an
I/O-efficient garbage collection scheme to reduce I/O overhead, (2) a
space-aware compaction strategy based on compensated size to mitigate
index-induced space amplification, and (3) a dynamic GC scheduler that adapts
to system load to make better use of CPU and storage resources. Extensive
experiments demonstrate that Scavenger+ significantly improves write
performance and reduces space amplification compared to state-of-the-art
KV-separated LSM-trees, including BlobDB, Titan, and TerarkDB.

### 3. [Query Logs Analytics: A Aystematic Literature Review](http://arxiv.org/pdf/2508.13949v1)

Authors: Dihia Lanasri

In the digital era, user interactions with various resources such as
databases, data warehouses, websites, and knowledge graphs (KGs) are
increasingly mediated through digital platforms. These interactions leave
behind digital traces, systematically captured in the form of logs. Logs, when
effectively exploited, provide high value across industry and academia,
supporting critical services (e.g., recovery and security), user-centric
applications (e.g., recommender systems), and quality-of-service improvements
(e.g., performance optimization). Despite their importance, research on log
usage remains fragmented across domains, and no comprehensive study currently
consolidates existing efforts. This paper presents a systematic survey of log
usage, focusing on Database (DB), Data Warehouse (DW), Web, and KG logs. More
than 300 publications were analyzed to address three central questions: (1) do
different types of logs share common structural and functional characteristics?
(2) are there standard pipelines for their usage? (3) which constraints and
non-functional requirements (NFRs) guide their exploitation?. The survey
reveals a limited number of end-to-end approaches, the absence of
standardization across log usage pipelines, and the existence of shared
structural elements among different types of logs. By consolidating existing
knowledge, identifying gaps, and highlighting opportunities, this survey
provides researchers and practitioners with a comprehensive overview of log
usage and sheds light on promising directions for future research, particularly
regarding the exploitation and democratization of KG logs.

### Distributed, Parallel, and Cluster Computing

### 1. [LUNDIsim: model meshes for flow simulation and scientific data compression benchmarks](http://arxiv.org/pdf/2508.13636v1)

Authors: Laurent Duval, Frédéric Payan, Christophe Preux, Lauriane Bouard

The volume of scientific data produced for and by numerical simulation
workflows is increasing at an incredible rate. This raises concerns either in
computability, interpretability, and sustainability. This is especially
noticeable in earth science (geology, meteorology, oceanography, and
astronomy), notably with climate studies.
  We highlight five main evaluation issues: efficiency, discrepancy, diversity,
interpretability, availability.
  Among remedies, lossless and lossy compression techniques are becoming
popular to better manage dataset volumes. Performance assessment -- with
comparative benchmarks -- require open datasets shared under FAIR principles
(Findable, Accessible, Interoperable, Reusable), with MRE (Minimal Reproducible
Example) ancillary data for reuse. We share LUNDIsim, an exemplary faulted
geological mesh. It is inspired by SPE10 comparative Challenge. Enhanced by
porosity/permeability datasets, this dataset proposes four distinct subsurface
environments. They were primarily designed for flow simulation in porous media.
Several consistent resolutions (with HexaShrink multiscale representations) are
proposed for each model. We also provide a set of reservoir features for
reproducing typical two-phase flow simulations on all LUNDIsim models in a
reservoir engineering context. This dataset is chiefly meant for benchmarking
and evaluating data size reduction (upscaling) or genuine composite mesh
compression algorithms. It is also suitable for other advanced mesh processing
workflows in geology and reservoir engineering, from visualization to machine
learning.
  LUNDIsim meshes are available at https://doi.org/10.5281/zenodo.14641958

### 2. [Estimating CO$_2$ emissions of distributed applications and platforms with SimGrid/Batsim](http://arxiv.org/pdf/2508.13693v1)

Authors: Gabriella Saraiva, Miguel Vasconcelos, Sarita Mazzini Bruschi, Danilo Carastan-Santos, Daniel Cordeiro

This work presents a carbon footprint plugin designed to extend the
capabilities of the Batsim simulator by allowing the calculation of CO$_2$
emissions during simulation runs. The goal is to comprehensively assess the
environmental impact associated with task and resource management strategies in
data centers. The plugin is developed within SimGrid -- the underlying
simulation framework of Batsim -- and computes carbon emissions based on the
simulated platform's energy consumption and carbon intensity factor of the
simulated machines. Once implemented, it is integrated into Batsim, ensuring
compatibility with existing simulation workflows and enabling researchers to
assess the carbon efficiency of their scheduling strategies.

### 3. [CaPGNN: Optimizing Parallel Graph Neural Network Training with Joint Caching and Resource-Aware Graph Partitioning](http://arxiv.org/pdf/2508.13716v1)

Authors: Xianfeng Song, Yi Zou, Zheng Shi

Graph Neural Networks (GNNs) have shown remarkable capabilities in processing
graph-structured data prevalent in various real-world applications. However,
the scalability of full-batch GNN training becomes severely limited by high
communication overhead and load imbalance in distributed environments. In this
paper, we present CaPGNN, a novel framework for efficient parallel full-batch
GNN training on single-server with multi-GPU, designed specifically to reduce
redundant inter-GPU communication and balance computational workloads. We
propose a joint adaptive caching algorithm that leverages both CPU and GPU
memory to significantly reduce the repetitive transmission of vertex features
across partitions. Additionally, we introduce a resource-aware graph
partitioning algorithm that adjusts subgraph sizes dynamically according to the
heterogeneous computational and communication capacities of GPUs. Extensive
experiments on large-scale benchmark datasets demonstrate that CaPGNN
effectively reduces communication costs by up to 96% and accelerates GNN
training by up to 12.7 times compared to state-of-the-art approaches. Our
results highlight the potential of adaptive caching and resource-aware
partitioning to facilitate scalable, efficient, and practical deployment of
full-batch GNN training in distributed computing environments.

### 4. [Is RISC-V ready for High Performance Computing? An evaluation of the Sophon SG2044](http://arxiv.org/pdf/2508.13840v1)

Authors: Nick Brown

The pace of RISC-V adoption continues to grow rapidly, yet for the successes
enjoyed in areas such as embedded computing, RISC-V is yet to gain ubiquity in
High Performance Computing (HPC). The Sophon SG2044 is SOPHGO's next generation
64-core high performance CPU that has been designed for workstation and server
grade workloads. Building upon the SG2042, subsystems that were a bottleneck in
the previous generation have been upgraded.
  In this paper we undertake the first performance study of the SG2044 for HPC.
Comparing against the SG2042 and other architectures, we find that the SG2044
is most advantageous when running at higher core counts, delivering up to 4.91
greater performance than the SG2042 over 64-cores. Two of the most important
upgrades in the SG2044 are support for RVV v1.0 and an enhanced memory
subsystem. This results in the SG2044 significantly closing the performance gap
with other architectures, especially for compute-bound workloads.

### 5. [Trans-XFed: An Explainable Federated Learning for Supply Chain Credit Assessment](http://arxiv.org/pdf/2508.13715v1)

Authors: Jie Shi, Arno P. J. M. Siebes, Siamak Mehrkanoon

This paper proposes a Trans-XFed architecture that combines federated
learning with explainable AI techniques for supply chain credit assessment. The
proposed model aims to address several key challenges, including privacy,
information silos, class imbalance, non-identically and independently
distributed (Non-IID) data, and model interpretability in supply chain credit
assessment. We introduce a performance-based client selection strategy (PBCS)
to tackle class imbalance and Non-IID problems. This strategy achieves faster
convergence by selecting clients with higher local F1 scores. The FedProx
architecture, enhanced with homomorphic encryption, is used as the core model,
and further incorporates a transformer encoder. The transformer encoder block
provides insights into the learned features. Additionally, we employ the
integrated gradient explainable AI technique to offer insights into
decision-making. We demonstrate the effectiveness of Trans-XFed through
experimental evaluations on real-world supply chain datasets. The obtained
results show its ability to deliver accurate credit assessments compared to
several baselines, while maintaining transparency and privacy.

### 6. [DDoS Attacks in Cloud Computing: Detection and Prevention](http://arxiv.org/pdf/2508.13522v1)

Authors: Zain Ahmad, Musab Ahmad, Bilal Ahmad

DDoS attacks are one of the most prevalent and harmful cybersecurity threats
faced by organizations and individuals today. In recent years, the complexity
and frequency of DDoS attacks have increased significantly, making it
challenging to detect and mitigate them effectively. The study analyzes various
types of DDoS attacks, including volumetric, protocol, and application layer
attacks, and discusses the characteristics, impact, and potential targets of
each type. It also examines the existing techniques used for DDoS attack
detection, such as packet filtering, intrusion detection systems, and machine
learning-based approaches, and their strengths and limitations. Moreover, the
study explores the prevention techniques employed to mitigate DDoS attacks,
such as firewalls, rate limiting , CPP and ELD mechanism. It evaluates the
effectiveness of each approach and its suitability for different types of
attacks and environments. In conclusion, this study provides a comprehensive
overview of the different types of DDoS attacks, their detection, and
prevention techniques. It aims to provide insights and guidelines for
organizations and individuals to enhance their cybersecurity posture and
protect against DDoS attacks.

### 7. [LAMMPS-KOKKOS: Performance Portable Molecular Dynamics Across Exascale Architectures](http://arxiv.org/pdf/2508.13523v1)

Authors: Anders Johansson, Evan Weinberg, Christian R. Trott, Megan J. McCarthy, Stan G. Moore

Since its inception in 1995, LAMMPS has grown to be a world-class molecular
dynamics code, with thousands of users, over one million lines of code, and
multi-scale simulation capabilities. We discuss how LAMMPS has adapted to the
modern heterogeneous computing landscape by integrating the Kokkos performance
portability library into the existing C++ code. We investigate performance
portability of simple pairwise, many-body reactive, and machine-learned
force-field interatomic potentials. We present results on GPUs across different
vendors and generations, and analyze performance trends, probing FLOPS
throughput, memory bandwidths, cache capabilities, and thread-atomic operation
performance. Finally, we demonstrate strong scaling on all current US exascale
machines -- OLCF Frontier, and ALCF Aurora, and NNSA El Capitan -- for the
three potentials.

### 8. [PennyLane-Lightning MPI: A massively scalable quantum circuit simulator based on distributed computing in CPU clusters](http://arxiv.org/pdf/2508.13615v1)

Authors: Ji-Hoon Kang, Hoon Ryu

Quantum circuit simulations play a critical role in bridging the gap between
theoretical quantum algorithms and their practical realization on physical
quantum hardware, yet they face computational challenges due to the exponential
growth of quantum state spaces with increasing qubit size. This work presents
PennyLane-Lightning MPI, an MPI-based extension of the PennyLane-Lightning
suite, developed to enable scalable quantum circuit simulations through
parallelization of quantum state vectors and gate operations across
distributed-memory systems. The core of this implementation is an
index-dependent, gate-specific parallelization strategy, which fully exploits
the characteristic of individual gates as well as the locality of computation
associated with qubit indices in partitioned state vectors. Benchmarking tests
with single gates and well-designed quantum circuits show that the present
method offers advantages in performance over general methods based on unitary
matrix operations and exhibits excellent scalability, supporting simulations of
up to 41-qubit with hundreds of thousands of parallel processes. Being equipped
with a Python plug-in for seamless integration to the PennyLane framework, this
work contributes to extending the PennyLane ecosystem by enabling
high-performance quantum simulations in standard multi-core CPU clusters with
no library-specific requirements, providing a back-end resource for the
cloud-based service framework of quantum computing that is under development in
the Republic of Korea.

### 9. [On the Security and Privacy of Federated Learning: A Survey with Attacks, Defenses, Frameworks, Applications, and Future Directions](http://arxiv.org/pdf/2508.13730v1)

Authors: Daniel M. Jimenez-Gutierrez, Yelizaveta Falkouskaya, Jose L. Hernandez-Ramos, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti

Federated Learning (FL) is an emerging distributed machine learning paradigm
enabling multiple clients to train a global model collaboratively without
sharing their raw data. While FL enhances data privacy by design, it remains
vulnerable to various security and privacy threats. This survey provides a
comprehensive overview of more than 200 papers regarding the state-of-the-art
attacks and defense mechanisms developed to address these challenges,
categorizing them into security-enhancing and privacy-preserving techniques.
Security-enhancing methods aim to improve FL robustness against malicious
behaviors such as byzantine attacks, poisoning, and Sybil attacks. At the same
time, privacy-preserving techniques focus on protecting sensitive data through
cryptographic approaches, differential privacy, and secure aggregation. We
critically analyze the strengths and limitations of existing methods, highlight
the trade-offs between privacy, security, and model performance, and discuss
the implications of non-IID data distributions on the effectiveness of these
defenses. Furthermore, we identify open research challenges and future
directions, including the need for scalable, adaptive, and energy-efficient
solutions operating in dynamic and heterogeneous FL environments. Our survey
aims to guide researchers and practitioners in developing robust and
privacy-preserving FL systems, fostering advancements safeguarding
collaborative learning frameworks' integrity and confidentiality.

### 10. [Analog computation with transcriptional networks](http://arxiv.org/pdf/2508.14017v1)

Authors: David Doty, Mina Latifi, David Soloveichick

Transcriptional networks represent one of the most extensively studied types
of systems in synthetic biology. Although the completeness of transcriptional
networks for digital logic is well-established, *analog* computation plays a
crucial role in biological systems and offers significant potential for
synthetic biology applications. While transcriptional circuits typically rely
on cooperativity and highly non-linear behavior of transcription factors to
regulate *production* of proteins, they are often modeled with simple linear
*degradation* terms. In contrast, general analog dynamics require both
non-linear positive as well as negative terms, seemingly necessitating control
over not just transcriptional (i.e., production) regulation but also the
degradation rates of transcription factors.
  Surprisingly, we prove that controlling transcription factor production
(i.e., transcription rate) without explicitly controlling degradation is
mathematically complete for analog computation, achieving equivalent
capabilities to systems where both production and degradation are programmable.
We demonstrate our approach on several examples including oscillatory and
chaotic dynamics, analog sorting, memory, PID controller, and analog extremum
seeking. Our result provides a systematic methodology for engineering novel
analog dynamics using synthetic transcriptional networks without the added
complexity of degradation control and informs our understanding of the
capabilities of natural transcriptional circuits.
  We provide a compiler, in the form of a Python package that can take any
system of polynomial ODEs and convert it to an equivalent transcriptional
network implementing the system *exactly*, under appropriate conditions.

### Discrete Mathematics

### 1. [Fair Division Among Couples and Small Groups](http://arxiv.org/pdf/2508.13432v1)

Authors: Paul Gölz, Hannane Yaghoubizade

We study the fair allocation of indivisible goods across groups of agents,
where each agent fully enjoys all goods allocated to their group. We focus on
groups of two (couples) and other groups of small size. For two couples, an EF1
allocation -- one in which all agents find their group's bundle no worse than
the other group's, up to one good -- always exists and can be found
efficiently. For three or more couples, EF1 allocations need not exist.
  Turning to proportionality, we show that, whenever groups have size at most
$k$, a PROP$k$ allocation exists and can be found efficiently. In fact, our
algorithm additionally guarantees (fractional) Pareto optimality, and PROP1 to
the first agent in each group, PROP2 to the second, etc., for an arbitrary
agent ordering. In special cases, we show that there are PROP1 allocations for
any number of couples.

### 2. [A Biased Random Key Genetic Algorithm for Solving the Longest Run Subsequence Problem](http://arxiv.org/pdf/2508.14020v1)

Authors: Christian Blum, Pedro Pinacho-Davidson

The longest run subsequence (LRS) problem is an NP-hard combinatorial
optimization problem belonging to the class of subsequence problems from
bioinformatics. In particular, the problem plays a role in genome reassembly.
In this paper, we present a solution to the LRS problem using a Biased Random
Key Genetic Algorithm (BRKGA). Our approach places particular focus on the
computational efficiency of evaluating individuals, which involves converting
vectors of gray values into valid solutions to the problem. For comparison
purposes, a Max-Min Ant System is developed and implemented. This is in
addition to the application of the integer linear programming solver CPLEX for
solving all considered problem instances. The computation results show that the
proposed BRKGA is currently a state-of-the-art technique for the LRS problem.
Nevertheless, the results also show that there is room for improvement,
especially in the context of input strings based on large alphabet sizes.

### Data Structures and Algorithms

### 1. [Generating the Spanning Trees of Series-Parallel Graphs up to Graph Automorphism](http://arxiv.org/pdf/2508.13480v1)

Authors: Mithra Karamchedu, Lucas Bang

In this paper, we investigate the problem of generating the spanning trees of
a graph $G$ up to the automorphisms or "symmetries" of $G$. After introducing
and surveying this problem for general input graphs, we present algorithms that
fully solve the case of series-parallel graphs, under two standard definitions.
We first show how to generate the nonequivalent spanning trees of a oriented
series-parallel graph $G$ in output-linear time, where both terminals of $G$
have been individually distinguished (i.e. applying an automorphism that
exchanges the terminals produces a different series-parallel graph).
Subsequently, we show how to adapt these oriented algorithms to the case of
semioriented series-parallel graphs, where we still have a set of two
distinguished terminals but neither has been designated as a source or sink.
Finally, we discuss the case of unoriented series-parallel graphs, where no
terminals have been distinguished and present a few observations and open
questions relating to them. The algorithms we present generate the
nonequivalent spanning trees of $G$ but never explicitly compute the
automorphism group of $G$, revealing how the recursive structure of $G$'s
automorphism group mirrors that of its spanning trees.

### 2. [Finding subdigraphs in digraphs of bounded directed treewidth](http://arxiv.org/pdf/2508.13830v1)

Authors: Raul Lopes, Ignasi Sau

It is well known that directed treewidth does not enjoy the nice algorithmic
properties of its undirected counterpart. There exist, however, some positive
results that, essentially, present XP algorithms for the problem of finding, in
a given digraph $D$, a subdigraph isomorphic to a digraph $H$ that can be
formed by the union of $k$ directed paths (with some extra properties),
parameterized by $k$ and the directed treewidth of $D$. Our motivation is to
tackle the following question: Are there subdigraphs, other than the directed
paths, that can be found efficiently in digraphs of bounded directed treewidth?
In a nutshell, the main message of this article is that, other than the
directed paths, the only digraphs that seem to behave well with respect to
directed treewidth are the stars. For this, we present a number of positive and
negative results, generalizing several results in the literature, as well as
some directions for further research.

### 3. [Online Stochastic Packing with General Correlations](http://arxiv.org/pdf/2508.13458v1)

Authors: Sabri Cetin, Yilun Chen, David A. Goldberg

There has been a growing interest in studying online stochastic packing under
more general correlation structures, motivated by the complex data sets and
models driving modern applications. Several past works either assume
correlations are weak or have a particular structure, have a complexity scaling
with the number of Markovian "states of the world" (which may be exponentially
large e.g. in the case of full history dependence), scale poorly with the
horizon $T$, or make additional continuity assumptions. Surprisingly, we show
that for all $\epsilon$, the online stochastic packing linear programming
problem with general correlations (suitably normalized and with sparse columns)
has an approximately optimal policy (with optimality gap $\epsilon T$) whose
per-decision runtime scales as the time to simulate a single sample path of the
underlying stochastic process (assuming access to a Monte Carlo simulator),
multiplied by a constant independent of the horizon or number of Markovian
states. We derive analogous results for network revenue management, and online
bipartite matching and independent set in bounded-degree graphs, by rounding.
Our algorithms implement stochastic gradient methods in a novel
on-the-fly/recursive manner for the associated massive deterministic-equivalent
linear program on the corresponding probability space.

### Emerging Technologies

### 1. [Quantum-Inspired Artificial Bee Colony for Latency-Aware Task Offloading in IoV](http://arxiv.org/pdf/2508.13637v1)

Authors: Mamta Kumari, Mayukh Sarkar, Rohit Kumar Nonia

Efficient task offloading is crucial for reducing latency and ensuring timely
decision-making in intelligent transportation systems within the rapidly
evolving Internet of Vehicles (IoV) landscape. This paper introduces a novel
Quantum-Inspired Artificial Bee Colony (QABC) algorithm specifically designed
for latency-sensitive task offloading involving cloud servers, Roadside Units
(RSUs), and vehicular nodes. By incorporating principles from quantum
computing, such as quantum state evolution and probabilistic encoding, QABC
enhances the classical Artificial Bee Colony (ABC) algorithm's ability to avoid
local optima and explore high-dimensional solution spaces. This research
highlights the potential of quantum-inspired heuristics to optimize real-time
offloading strategies in future vehicular networks.

### 2. [Virtuous Machines: Towards Artificial General Science](http://arxiv.org/pdf/2508.13421v1)

Authors: Gabrielle Wehr, Reuben Rideaux, Amaya J. Fox, David R. Lightfoot, Jason Tangen, Jason B. Mattingley, Shane E. Ehrhardt

Artificial intelligence systems are transforming scientific discovery by
accelerating specific research tasks, from protein structure prediction to
materials design, yet remain confined to narrow domains requiring substantial
human oversight. The exponential growth of scientific literature and increasing
domain specialisation constrain researchers' capacity to synthesise knowledge
across disciplines and develop unifying theories, motivating exploration of
more general-purpose AI systems for science. Here we show that a
domain-agnostic, agentic AI system can independently navigate the scientific
workflow - from hypothesis generation through data collection to manuscript
preparation. The system autonomously designed and executed three psychological
studies on visual working memory, mental rotation, and imagery vividness,
executed one new online data collection with 288 participants, developed
analysis pipelines through 8-hour+ continuous coding sessions, and produced
completed manuscripts. The results demonstrate the capability of AI scientific
discovery pipelines to conduct non-trivial research with theoretical reasoning
and methodological rigour comparable to experienced researchers, though with
limitations in conceptual nuance and theoretical interpretation. This is a step
toward embodied AI that can test hypotheses through real-world experiments,
accelerating discovery by autonomously exploring regions of scientific space
that human cognitive and resource constraints might otherwise leave unexplored.
It raises important questions about the nature of scientific understanding and
the attribution of scientific credit.

### 3. [A fully-programmable integrated photonic processor for both domain-specific and general-purpose computing](http://arxiv.org/pdf/2508.13551v1)

Authors: Feng-Kai Han, Xiao-Yun Xu, Tian-Yu Zhang, Lei Feng, Chu-Han Wang, Jie Ma, Ze-Feng Lan, Chao-Qian Li, Yi Xie, Hai Yan, Yu-Fei Liu, Yu-Quan Peng, Xian-Min Jin

A variety of complicated computational scenarios have made unprecedented
demands on the computing power and energy efficiency of electronic computing
systems, including solving intractable nondeterministic polynomial-time
(NP)-complete problems and dealing with large-scale artificial intelligence
models. Optical computing emerges as a promising paradigm to meet these
challenges, whereas current optical computing architectures have limited
versatility. Their applications are usually either constrained to a specialized
domain or restricted to general-purpose matrix computation. Here, we implement
a fully-programmable integrated photonic processor that can be configured to
tackle both specific computational problems and general-purpose matrix
computation. We achieve complete end-to-end control of the photonic processor
by utilizing a self-developed integrated programmable optoelectronic computing
platform. For domain-specific computing, our photonic processor can efficiently
solve two kinds of NP-complete problems: subset sum problem (far more than 2^N
different instances) and exact cover problem. For general-purpose computation,
we experimentally demonstrate high-precision optical dot product and further
realize accurate image edge detection and MNIST handwritten image
classification task with an accuracy of 97%. Our work enhances the versatility
and capability of optical computing architecture, paving the way for its
practical application in future high-performance and complex computing
scenarios.

### 4. [Security-as-a-Function for IDS/IPS in Softwarized Network and Applications to 5G Network Systems](http://arxiv.org/pdf/2508.13581v1)

Authors: Shivank Malik, Samaresh Bera

The service-based architecture of 5G network allows network operators to
place virtualized network functions on commodity hardware, unlike the
traditional vendor-specific hardware-based functionalities. However, it expands
the security vulnerabilities and threats to the 5G network. While there exist
several theoretical studies on network function placement and service routing,
a few focused on the security aspects of the 5G network systems.
  This paper focuses on safeguarding the 5G core network systems from DoS and
DDoS attacks by placing intrusion detection and prevention systems (IDS-IPS) as
virtualized network functions following the 5G standalone architecture. To
ensure the virtualized placement of IDS-IPS, first, we provide thorough virtual
machine (VM)-based and containerized implementation details and evaluate the
network performance with two scenarios, IDS and IPS, in the presence of TCP and
UDP applications. Second, we apply the VM-based implementation of IDS-IPS on a
softwarized 5G core network and study the network performances. The experiment
results on network throughput, latency, and packet drop reveal that the
softwarized IDS-IPS can meet the QoS requirements of 5G applications, while
safeguarding the network from DoS and DDoS attacks.

### 5. [Portfolio construction using a sampling-based variational quantum scheme](http://arxiv.org/pdf/2508.13557v1)

Authors: Gabriele Agliardi, Dimitris Alevras, Vaibhaw Kumar, Roberto Lo Nardo, Gabriele Compostella, Sumit Kumar, Manuel Proissl, Bimal Mehta

The efficient and effective construction of portfolios that adhere to
real-world constraints is a challenging optimization task in finance. We
investigate a concrete representation of the problem with a focus on design
proposals of an Exchange Traded Fund. We evaluate the sampling-based CVaR
Variational Quantum Algorithm (VQA), combined with a local-search
post-processing, for solving problem instances that beyond a certain size
become classically hard. We also propose a problem formulation that is suited
for sampling-based VQA. Our utility-scale experiments on IBM Heron processors
involve 109 qubits and up to 4200 gates, achieving a relative solution error of
0.49%. Results indicate that a combined quantum-classical workflow achieves
better accuracy compared to purely classical local search, and that
hard-to-simulate quantum circuits may lead to better convergence than simpler
circuits. Our work paves the path to further explore portfolio construction
with quantum computers.

### 6. [Analog computation with transcriptional networks](http://arxiv.org/pdf/2508.14017v1)

Authors: David Doty, Mina Latifi, David Soloveichick

Transcriptional networks represent one of the most extensively studied types
of systems in synthetic biology. Although the completeness of transcriptional
networks for digital logic is well-established, *analog* computation plays a
crucial role in biological systems and offers significant potential for
synthetic biology applications. While transcriptional circuits typically rely
on cooperativity and highly non-linear behavior of transcription factors to
regulate *production* of proteins, they are often modeled with simple linear
*degradation* terms. In contrast, general analog dynamics require both
non-linear positive as well as negative terms, seemingly necessitating control
over not just transcriptional (i.e., production) regulation but also the
degradation rates of transcription factors.
  Surprisingly, we prove that controlling transcription factor production
(i.e., transcription rate) without explicitly controlling degradation is
mathematically complete for analog computation, achieving equivalent
capabilities to systems where both production and degradation are programmable.
We demonstrate our approach on several examples including oscillatory and
chaotic dynamics, analog sorting, memory, PID controller, and analog extremum
seeking. Our result provides a systematic methodology for engineering novel
analog dynamics using synthetic transcriptional networks without the added
complexity of degradation control and informs our understanding of the
capabilities of natural transcriptional circuits.
  We provide a compiler, in the form of a Python package that can take any
system of polynomial ODEs and convert it to an equivalent transcriptional
network implementing the system *exactly*, under appropriate conditions.

### Formal Languages and Automata Theory

### 1. [Programmable Anyon Mobility through Higher Order Cellular Automata](http://arxiv.org/pdf/2508.13961v1)

Authors: Jie-Yu Zhang, Peng Ye

Controlling anyon mobility is critical for robust quantum memory and
understanding symmetry-enriched topological (SET) phases with subsystem
symmetries (e.g., line-like, fractal, chaotic, or mixed supports). However, a
unified framework for anyon mobility in SET phases with such diverse geometric
patterns of symmetry supports has remained a major challenge. In this Letter,
by introducing higher-order cellular automata (HOCA) -- a powerful computer
science tool -- to SET physics, we establish a unified approach for complete
characterization of anyon mobility induced by the complexity of subsystem
symmetries. First, we design finite-depth HOCA-controlled unitary quantum
circuits, yielding exactly solvable SET models with Abelian anyons and all
possible locally generated subsystem symmetries. Then, we present a theorem
that precisely programs all excitation mobilities (fractons, lineons, or fully
mobile anyons) directly from the HOCA rule, representing the first complete
characterization of anyon mobility in SET phases. As a corollary, this theorem
yields symmetry-enriched fusion rules which govern mobility transmutation
during fusion. Fusion rules with multiple channels are identified, exhibiting
non-Abelian characteristics in Abelian anyon systems. Leveraging HOCA, this
Letter opens new avenues for characterization of SET phases of matter and
programmability of topological quantum codes.

### Graphics

### 1. [Eliminating Rasterization: Direct Vector Floor Plan Generation with DiffPlanner](http://arxiv.org/pdf/2508.13738v1)

Authors: Shidong Wang, Renato Pajarola

The boundary-constrained floor plan generation problem aims to generate the
topological and geometric properties of a set of rooms within a given boundary.
Recently, learning-based methods have made significant progress in generating
realistic floor plans. However, these methods involve a workflow of converting
vector data into raster images, using image-based generative models, and then
converting the results back into vector data. This process is complex and
redundant, often resulting in information loss. Raster images, unlike vector
data, cannot scale without losing detail and precision. To address these
issues, we propose a novel deep learning framework called DiffPlanner for
boundary-constrained floor plan generation, which operates entirely in vector
space. Our framework is a Transformer-based conditional diffusion model that
integrates an alignment mechanism in training, aligning the optimization
trajectory of the model with the iterative design processes of designers. This
enables our model to handle complex vector data, better fit the distribution of
the predicted targets, accomplish the challenging task of floor plan layout
design, and achieve user-controllable generation. We conduct quantitative
comparisons, qualitative evaluations, ablation experiments, and perceptual
studies to evaluate our method. Extensive experiments demonstrate that
DiffPlanner surpasses existing state-of-the-art methods in generating floor
plans and bubble diagrams in the creative stages, offering more controllability
to users and producing higher-quality results that closely match the ground
truths.

### 2. [Sketch3DVE: Sketch-based 3D-Aware Scene Video Editing](http://arxiv.org/pdf/2508.13797v1)

Authors: Feng-Lin Liu, Shi-Yang Li, Yan-Pei Cao, Hongbo Fu, Lin Gao

Recent video editing methods achieve attractive results in style transfer or
appearance modification. However, editing the structural content of 3D scenes
in videos remains challenging, particularly when dealing with significant
viewpoint changes, such as large camera rotations or zooms. Key challenges
include generating novel view content that remains consistent with the original
video, preserving unedited regions, and translating sparse 2D inputs into
realistic 3D video outputs. To address these issues, we propose Sketch3DVE, a
sketch-based 3D-aware video editing method to enable detailed local
manipulation of videos with significant viewpoint changes. To solve the
challenge posed by sparse inputs, we employ image editing methods to generate
edited results for the first frame, which are then propagated to the remaining
frames of the video. We utilize sketching as an interaction tool for precise
geometry control, while other mask-based image editing methods are also
supported. To handle viewpoint changes, we perform a detailed analysis and
manipulation of the 3D information in the video. Specifically, we utilize a
dense stereo method to estimate a point cloud and the camera parameters of the
input video. We then propose a point cloud editing approach that uses depth
maps to represent the 3D geometry of newly edited components, aligning them
effectively with the original 3D scene. To seamlessly merge the newly edited
content with the original video while preserving the features of unedited
regions, we introduce a 3D-aware mask propagation strategy and employ a video
diffusion model to produce realistic edited videos. Extensive experiments
demonstrate the superiority of Sketch3DVE in video editing. Homepage and code:
http://http://geometrylearning.com/Sketch3DVE/

### 3. [Is-NeRF: In-scattering Neural Radiance Field for Blurred Images](http://arxiv.org/pdf/2508.13808v1)

Authors: Nan Luo, Chenglin Ye, Jiaxu Li, Gang Liu, Bo Wan, Di Wang, Lupeng Liu, Jun Xiao

Neural Radiance Fields (NeRF) has gained significant attention for its
prominent implicit 3D representation and realistic novel view synthesis
capabilities. Available works unexceptionally employ straight-line volume
rendering, which struggles to handle sophisticated lightpath scenarios and
introduces geometric ambiguities during training, particularly evident when
processing motion-blurred images. To address these challenges, this work
proposes a novel deblur neural radiance field, Is-NeRF, featuring explicit
lightpath modeling in real-world environments. By unifying six common light
propagation phenomena through an in-scattering representation, we establish a
new scattering-aware volume rendering pipeline adaptable to complex lightpaths.
Additionally, we introduce an adaptive learning strategy that enables
autonomous determining of scattering directions and sampling intervals to
capture finer object details. The proposed network jointly optimizes NeRF
parameters, scattering parameters, and camera motions to recover fine-grained
scene representations from blurry images. Comprehensive evaluations demonstrate
that it effectively handles complex real-world scenarios, outperforming
state-of-the-art approaches in generating high-fidelity images with accurate
geometric details.

### 4. [Uncertainty-Aware PCA for Arbitrarily Distributed Data Modeled by Gaussian Mixture Models](http://arxiv.org/pdf/2508.13990v1)

Authors: Daniel Klötzl, Ozan Tastekin, David Hägele, Marina Evers, Daniel Weiskopf

Multidimensional data is often associated with uncertainties that are not
well-described by normal distributions. In this work, we describe how such
distributions can be projected to a low-dimensional space using
uncertainty-aware principal component analysis (UAPCA). We propose to model
multidimensional distributions using Gaussian mixture models (GMMs) and derive
the projection from a general formulation that allows projecting arbitrary
probability density functions. The low-dimensional projections of the densities
exhibit more details about the distributions and represent them more faithfully
compared to UAPCA mappings. Further, we support including user-defined weights
between the different distributions, which allows for varying the importance of
the multidimensional distributions. We evaluate our approach by comparing the
distributions in low-dimensional space obtained by our method and UAPCA to
those obtained by sample-based projections.

### Computer Science and Game Theory

### 1. [Reactive Users vs. Recommendation Systems: An Adaptive Policy to Manage Opinion Drifts](http://arxiv.org/pdf/2508.13473v1)

Authors: Atefeh Mollabagher, Parinaz Naghizadeh

Recommendation systems are used in a range of platforms to maximize user
engagement through personalization and the promotion of popular content. It has
been found that such recommendations may shape users' opinions over time. In
this paper, we ask whether reactive users, who are cognizant of the influence
of the content they consume, can prevent such changes by adaptively adjusting
their content consumption choices. To this end, we study users' opinion
dynamics under two types of stochastic policies: a passive policy where the
probability of clicking on recommended content is fixed and a reactive policy
where clicking probability adaptively decreases following large opinion drifts.
We analytically derive the expected opinion and user utility under these
policies. We show that the adaptive policy can help users prevent opinion
drifts and that when a user prioritizes opinion preservation, the expected
utility of the adaptive policy outperforms the fixed policy. We validate our
theoretical findings through numerical simulations. These findings help better
understand how user-level strategies can challenge the biases induced by
recommendation systems.

### 2. [Optimal Candidate Positioning in Multi-Issue Elections](http://arxiv.org/pdf/2508.13841v1)

Authors: Colin Cleveland, Bart de Keijzer, Maria Polukarov

We study strategic candidate positioning in multidimensional spatial-voting
elections. Voters and candidates are represented as points in $\mathbb{R}^d$,
and each voter supports the candidate that is closest under a distance induced
by an $\ell_p$-norm. We prove that computing an optimal location for a new
candidate is NP-hard already against a single opponent, whereas for a constant
number of issues the problem is tractable: an $O(n^{d+1})$
hyperplane-enumeration algorithm and an $O(n \log n)$ radial-sweep routine for
$d=2$ solve the task exactly. We further derive the first approximation
guarantees for the general multi-candidate case and show how our geometric
approach extends seamlessly to positional-scoring rules such as $k$-approval
and Borda. These results clarify the algorithmic landscape of multidimensional
spatial elections and provide practically implementable tools for campaign
strategy.

### 3. [Control by Deleting Players from Weighted Voting Games Is NP^PP-Complete for the Penrose-Banzhaf Power Index](http://arxiv.org/pdf/2508.13868v1)

Authors: Joanna Kaczmarek, Jörg Rothe

Weighted voting games are a popular class of coalitional games that are
widely used to model real-life situations of decision-making. They can be
applied, for instance, to analyze legislative processes in parliaments or
voting in corporate structures. Various ways of tampering with these games have
been studied, among them merging or splitting players, fiddling with the quota,
and controlling weighted voting games by adding or deleting players. While the
complexity of control by adding players to such games so as to change or
maintain a given player's power has been recently settled, the complexity of
control by deleting players from such games (with the same goals) remained
open. We show that when the players' power is measured by the probabilistic
Penrose-Banzhaf index, some of these problems are complete for NP^PP -- the
class of problems solvable by NP machines equipped with a PP ("probabilistic
polynomial time") oracle. Our results optimally improve the currently known
lower bounds of hardness for much smaller complexity classes, thus providing
protection against SAT-solving techniques in practical applications.

### 4. [Fair Division Among Couples and Small Groups](http://arxiv.org/pdf/2508.13432v1)

Authors: Paul Gölz, Hannane Yaghoubizade

We study the fair allocation of indivisible goods across groups of agents,
where each agent fully enjoys all goods allocated to their group. We focus on
groups of two (couples) and other groups of small size. For two couples, an EF1
allocation -- one in which all agents find their group's bundle no worse than
the other group's, up to one good -- always exists and can be found
efficiently. For three or more couples, EF1 allocations need not exist.
  Turning to proportionality, we show that, whenever groups have size at most
$k$, a PROP$k$ allocation exists and can be found efficiently. In fact, our
algorithm additionally guarantees (fractional) Pareto optimality, and PROP1 to
the first agent in each group, PROP2 to the second, etc., for an arbitrary
agent ordering. In special cases, we show that there are PROP1 allocations for
any number of couples.

### 5. [When Does Selfishness Align with Team Goals? A Structural Analysis of Equilibrium and Optimality](http://arxiv.org/pdf/2508.13450v1)

Authors: Gehui Xu, Thomas Parisini, Andreas A. Malikopoulos

This paper investigates the relationship between the team-optimal solution
and the Nash equilibrium (NE) to assess the impact of self-interested decisions
on team performance. In classical team decision problems, team members
typically act cooperatively towards a common objective to achieve a
team-optimal solution. However, in practice, members may behave selfishly by
prioritizing their goals, resulting in an NE under a non-cooperative game. To
study this misalignment, we develop a parameterized model for team and game
problems, where game parameters represent each individual's deviation from the
team objective. The study begins by exploring the consistency and deviation
between the NE and the team-optimal solution under fixed game parameters. We
provide a necessary and sufficient condition for any NE to be a team optimum,
along with establishing an upper bound to measure their difference when the
consistency condition fails. The exploration then focuses on aligning NE
strategies towards the team-optimal solution through the adjustment of game
parameters, resulting in a non-convex and non-smooth bi-level optimization
problem. We propose a hypergradient-based algorithm for this problem, and
establish its convergence to the critical points. Finally, we validate our
theoretical findings through extensive simulation studies.

### 6. [The Multi-Stage Assignment Problem: A Fairness Perspective](http://arxiv.org/pdf/2508.13856v1)

Authors: Vibulan J, Swapnil Dhamal, Shweta Jain

This paper explores the problem of fair assignment on Multi-Stage graphs. A
multi-stage graph consists of nodes partitioned into $K$ disjoint sets (stages)
structured as a sequence of weighted bipartite graphs formed across adjacent
stages. The goal is to assign node-disjoint paths to $n$ agents starting from
the first stage and ending in the last stage. We show that an efficient
assignment that minimizes the overall sum of costs of all the agents' paths may
be highly unfair and lead to significant cost disparities (envy) among the
agents. We further show that finding an envy-minimizing assignment on a
multi-stage graph is NP-hard. We propose the C-Balance algorithm, which
guarantees envy that is bounded by $2M$ in the case of two agents, where $M$ is
the maximum edge weight. We demonstrate the algorithm's tightness by presenting
an instance where the envy is $2M$. We further show that the cost of fairness
($CoF$), defined as the ratio of the cost of the assignment given by the fair
algorithm to that of the minimum cost assignment, is bounded by $2$ for
C-Balance. We then extend this approach to $n$ agents by proposing the
DC-Balance algorithm that makes iterative calls to C-Balance. We show the
convergence of DC-Balance, resulting in envy that is arbitrarily close to $2M$.
We derive $CoF$ bounds for DC-Balance and provide insights about its dependency
on the instance-specific parameters and the desired degree of envy. We
experimentally show that our algorithm runs several orders of magnitude faster
than a suitably formulated ILP.

### 7. [A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources -- Extended Version](http://arxiv.org/pdf/2508.13960v1)

Authors: Björn Filter, Ralf Möller, Özgür Lütfü Özçep

The latest developments in AI focus on agentic systems where artificial and
human agents cooperate to realize global goals. An example is collaborative
learning, which aims to train a global model based on data from individual
agents. A major challenge in designing such systems is to guarantee safety and
alignment with human values, particularly a fair distribution of rewards upon
achieving the global goal. Cooperative game theory offers useful abstractions
of cooperating agents via value functions, which assign value to each
coalition, and via reward functions. With these, the idea of fair allocation
can be formalized by specifying fairness axioms and designing concrete
mechanisms. Classical cooperative game theory, exemplified by the Shapley
value, does not fully capture scenarios like collaborative learning, as it
assumes nonreplicable resources, whereas data and models can be replicated.
Infinite replicability requires a generalized notion of fairness, formalized
through new axioms and mechanisms. These must address imbalances in reciprocal
benefits among participants, which can lead to strategic exploitation and
unfair allocations. The main contribution of this paper is a mechanism and a
proof that it fulfills the property of mutual fairness, formalized by the
Balanced Reciprocity Axiom. It ensures that, for every pair of players, each
benefits equally from the participation of the other.

### Human-Computer Interaction

### 1. [Visuo-Tactile Feedback with Hand Outline Styles for Modulating Affective Roughness Perception](http://arxiv.org/pdf/2508.13504v1)

Authors: Minju Baeck, Yoonseok Shin, Dooyoung Kim, Hyunjin Lee, Sang Ho Yoon, Woontack Woo

We propose a visuo-tactile feedback method that combines virtual hand
visualization and fingertip vibrations to modulate affective roughness
perception in VR. While prior work has focused on object-based textures and
vibrotactile feedback, the role of visual feedback on virtual hands remains
underexplored. Our approach introduces affective visual cues including line
shape, motion, and color applied to hand outlines, and examines their influence
on both affective responses (arousal, valence) and perceived roughness. Results
show that sharp contours enhanced perceived roughness, increased arousal, and
reduced valence, intensifying the emotional impact of haptic feedback. In
contrast, color affected valence only, with red consistently lowering emotional
positivity. These effects were especially noticeable at lower haptic
intensities, where visual cues extended affective modulation into mid-level
perceptual ranges. Overall, the findings highlight how integrating expressive
visual cues with tactile feedback can enrich affective rendering and offer
flexible emotional tuning in immersive VR interactions.

### 2. [koboshi: A Base That Animates Everyday Objects](http://arxiv.org/pdf/2508.13509v1)

Authors: Yuta Sugiura

We propose a base-shaped robot named "koboshi" that moves everyday objects.
This koboshi has a spherical surface in contact with the floor, and by moving a
weight inside using built-in motors, it can rock up and down, and side to side.
By placing everyday items on this koboshi, users can impart new movement to
otherwise static objects. The koboshi is equipped with sensors to measure its
posture, enabling interaction with users. Additionally, it has communication
capabilities, allowing multiple units to communicate with each other.

### 3. ["Can You See Me Think?" Grounding LLM Feedback in Keystrokes and Revision Patterns](http://arxiv.org/pdf/2508.13543v1)

Authors: Samra Zafar, Shifa Yousaf, Muhammad Shaheer Minhas

As large language models (LLMs) increasingly assist in evaluating student
writing, researchers have begun questioning whether these models can be
cognitively grounded, that is, whether they can attend not just to the final
product, but to the process by which it was written. In this study, we explore
how incorporating writing process data, specifically keylogs and time-stamped
snapshots, affects the quality of LLM-generated feedback. We conduct an
ablation study on 52 student essays comparing feedback generated with access to
only the final essay (C1) and feedback that also incorporates keylogs and
time-stamped snapshots (C2). While rubric scores changed minimally, C2 feedback
demonstrated significantly improved structural evaluation and greater
process-sensitive justification.

### 4. [`My Dataset of Love': A Preliminary Mixed-Method Exploration of Human-AI Romantic Relationships](http://arxiv.org/pdf/2508.13655v1)

Authors: Xuetong Wang, Ching Christie Pang, Pan Hui

Human-AI romantic relationships have gained wide popularity among social
media users in China. The technological impact on romantic relationships and
its potential applications have long drawn research attention to topics such as
relationship preservation and negativity mitigation. Media and communication
studies also explore the practices in romantic para-social relationships.
Nonetheless, this emerging human-AI romantic relationship, whether the
relations fall into the category of para-social relationship together with its
navigation pattern, remains unexplored, particularly in the context of
relational stages and emotional attachment. This research thus seeks to fill
this gap by presenting a mixed-method approach on 1,766 posts and 60,925
comments from Xiaohongshu, as well as the semi-structured interviews with 23
participants, of whom one of them developed her relationship with self-created
AI for three years. The findings revealed that the users' willingness to
self-disclose to AI companions led to increased positivity without social
stigma. The results also unveiled the reciprocal nature of these interactions,
the dominance of 'self', and raised concerns about language misuse, bias, and
data security in AI communication.

### 5. [Bend It, Aim It, Tap It: Designing an On-Body Disambiguation Mechanism for Curve Selection in Mixed Reality](http://arxiv.org/pdf/2508.13748v1)

Authors: Xiang Li, Per Ola Kristensson

Object selection in Mixed Reality (MR) becomes particularly challenging in
dense or occluded environments, where traditional mid-air ray-casting often
leads to ambiguity and reduced precision. We present two complementary
techniques: (1) a real-time Bezier Curve selection paradigm guided by finger
curvature, enabling expressive one-handed trajectories, and (2) an on-body
disambiguation mechanism that projects the four nearest candidates onto the
user's forearm via proximity-based mapping. Together, these techniques combine
flexible, user-controlled selection with tactile, proprioceptive
disambiguation. We evaluated their independent and joint effects in a 2x2
within-subjects study (N = 24), crossing interaction paradigm (Bezier Curve vs.
Linear Ray) with interaction medium (Mid-air vs. On-body). Results show that
on-body disambiguation significantly reduced selection errors and physical
demand while improving perceived performance, hedonic quality, and user
preference. Bezier input provided effective access to occluded targets but
incurred longer task times and greater effort under some conditions. We
conclude with design implications for integrating curved input and on-body
previews to support precise, adaptive selection in immersive environments.

### 6. [Mind & Motion: Opportunities and Applications of Integrating Biomechanics and Cognitive Models in HCI](http://arxiv.org/pdf/2508.13788v1)

Authors: Arthur Fleig, Florian Fischer, Markus Klar, Patrick Ebel, Miroslav Bachinski, Per Ola Kristensson, Roderick Murray-Smith, Antti Oulasvirta

Computational models of how users perceive and act within a virtual or
physical environment offer enormous potential for the understanding and design
of user interactions. Cognition models have been used to understand the role of
attention and individual preferences and beliefs on human decision making
during interaction, while biomechanical simulations have been successfully
applied to analyse and predict physical effort, fatigue, and discomfort. The
next frontier in HCI lies in connecting these models to enable robust, diverse,
and representative simulations of different user groups. These embodied user
simulations could predict user intents, strategies, and movements during
interaction more accurately, benchmark interfaces and interaction techniques in
terms of performance and ergonomics, and guide adaptive system design. This
UIST workshop explores ideas for integrating computational models into HCI and
discusses use cases such as UI/UX design, automated system testing, and
personalised adaptive interfaces. It brings researchers from relevant
disciplines together to identify key opportunities and challenges as well as
feasible next steps for bridging mind and motion to simulate interactive user
behaviour.

### 7. [Large Language Models as Visualization Agents for Immersive Binary Reverse Engineering](http://arxiv.org/pdf/2508.13413v1)

Authors: Dennis Brown, Samuel Mulder

Immersive virtual reality (VR) offers affordances that may reduce cognitive
complexity in binary reverse engineering (RE), enabling embodied and external
cognition to augment the RE process through enhancing memory, hypothesis
testing, and visual organization. In prior work, we applied a cognitive systems
engineering approach to identify an initial set of affordances and implemented
a VR environment to support RE through spatial persistence and interactivity.
In this work, we extend that platform with an integrated large language model
(LLM) agent capable of querying binary analysis tools, answering technical
questions, and dynamically generating immersive 3D visualizations in alignment
with analyst tasks. We describe the system architecture and our evaluation
process and results. Our pilot study shows that while LLMs can generate
meaningful 3D call graphs (for small programs) that align with design
principles, output quality varies widely. This work raises open questions about
the potential for LLMs to function as visualization agents, constructing 3D
representations that reflect cognitive design principles without explicit
training.

### 8. [Uncertainty Tube Visualization of Particle Trajectories](http://arxiv.org/pdf/2508.13505v1)

Authors: Jixian Li, Timbwaoga Aime Judicael Ouermi, Mengjiao Han, Chris R. Johnson

Predicting particle trajectories with neural networks (NNs) has substantially
enhanced many scientific and engineering domains. However, effectively
quantifying and visualizing the inherent uncertainty in predictions remains
challenging. Without an understanding of the uncertainty, the reliability of NN
models in applications where trustworthiness is paramount is significantly
compromised. This paper introduces the uncertainty tube, a novel,
computationally efficient visualization method designed to represent this
uncertainty in NN-derived particle paths. Our key innovation is the design and
implementation of a superelliptical tube that accurately captures and
intuitively conveys nonsymmetric uncertainty. By integrating well-established
uncertainty quantification techniques, such as Deep Ensembles, Monte Carlo
Dropout (MC Dropout), and Stochastic Weight Averaging-Gaussian (SWAG), we
demonstrate the practical utility of the uncertainty tube, showcasing its
application on both synthetic and simulation datasets.

### 9. [Beyond Human Judgment: A Bayesian Evaluation of LLMs' Moral Values Understanding](http://arxiv.org/pdf/2508.13804v1)

Authors: Maciej Skorski, Alina Landowska

How do large language models understand moral dimensions compared to humans?
  This first large-scale Bayesian evaluation of market-leading language models
provides the answer. In contrast to prior work using deterministic ground truth
(majority or inclusion rules), we model annotator disagreements to capture both
aleatoric uncertainty (inherent human disagreement) and epistemic uncertainty
(model domain sensitivity). We evaluate top language models (Claude Sonnet 4,
DeepSeek-V3, Llama 4 Maverick) across 250K+ annotations from ~700 annotators on
100K+ texts spanning social media, news, and forums.
  Our GPU-optimized Bayesian framework processed 1M+ model queries, revealing
that AI models typically rank among the top 25\% of human annotators, achieving
much better-than-average balanced accuracy. Importantly, we find that AI
produces far fewer false negatives than humans, highlighting their more
sensitive moral detection capabilities.

### 10. [LLM-Powered Virtual Patient Agents for Interactive Clinical Skills Training with Automated Feedback](http://arxiv.org/pdf/2508.13943v1)

Authors: Henrik Voigt, Yurina Sugamiya, Kai Lawonn, Sina Zarrieß, Atsuo Takanishi

Objective Structured Clinical Examinations (OSCEs) are essential for medical
training, but they require significant resources, including professional actors
and expert medical feedback. Although Large Language Models (LLMs) have
introduced text-based virtual patients for communication practice, these
simulations often lack the capability for richer, non-textual interactions.
This paper presents a novel framework that significantly enhances LLM-based
simulated patients by equipping them with action spaces, thereby enabling more
realistic and dynamic patient behaviors that extend beyond text. Furthermore,
our system incorporates virtual tutors that provide students with instant,
personalized feedback on their performance at any time during these simulated
encounters. We have conducted a rigorous evaluation of the framework's
real-time performance, including system latency and component accuracy.
Preliminary evaluations with medical experts assessed the naturalness and
coherence of the simulated patients, as well as the usefulness and
appropriateness of the virtual tutor's assessments. This innovative system
provides medical students with a low-cost, accessible platform for personalized
OSCE preparation at home.

### Information Retrieval

### 1. [ENCODE: Breaking the Trade-Off Between Performance and Efficiency in Long-Term User Behavior Modeling](http://arxiv.org/pdf/2508.13567v1)

Authors: Wenji Zhou, Yuhang Zheng, Yinfu Feng, Yunan Ye, Rong Xiao, Long Chen, Xiaosong Yang, Jun Xiao

Long-term user behavior sequences are a goldmine for businesses to explore
users' interests to improve Click-Through Rate. However, it is very challenging
to accurately capture users' long-term interests from their long-term behavior
sequences and give quick responses from the online serving systems. To meet
such requirements, existing methods "inadvertently" destroy two basic
requirements in long-term sequence modeling: R1) make full use of the entire
sequence to keep the information as much as possible; R2) extract information
from the most relevant behaviors to keep high relevance between learned
interests and current target items. The performance of online serving systems
is significantly affected by incomplete and inaccurate user interest
information obtained by existing methods. To this end, we propose an efficient
two-stage long-term sequence modeling approach, named as EfficieNt Clustering
based twO-stage interest moDEling (ENCODE), consisting of offline extraction
stage and online inference stage. It not only meets the aforementioned two
basic requirements but also achieves a desirable balance between online service
efficiency and precision. Specifically, in the offline extraction stage, ENCODE
clusters the entire behavior sequence and extracts accurate interests. To
reduce the overhead of the clustering process, we design a metric
learning-based dimension reduction algorithm that preserves the relative
pairwise distances of behaviors in the new feature space. While in the online
inference stage, ENCODE takes the off-the-shelf user interests to predict the
associations with target items. Besides, to further ensure the relevance
between user interests and target items, we adopt the same relevance metric
throughout the whole pipeline of ENCODE. The extensive experiment and
comparison with SOTA have demonstrated the effectiveness and efficiency of our
proposed ENCODE.

### 2. [MUFFIN: Mixture of User-Adaptive Frequency Filtering for Sequential Recommendation](http://arxiv.org/pdf/2508.13670v1)

Authors: Ilwoong Baek, Mincheol Yoon, Seongmin Park, Jongwuk Lee

Sequential recommendation (SR) aims to predict users' subsequent interactions
by modeling their sequential behaviors. Recent studies have explored frequency
domain analysis, which effectively models periodic patterns in user sequences.
However, existing frequency-domain SR models still face two major drawbacks:
(i) limited frequency band coverage, often missing critical behavioral patterns
in a specific frequency range, and (ii) lack of personalized frequency
filtering, as they apply an identical filter for all users regardless of their
distinct frequency characteristics. To address these challenges, we propose a
novel frequency-domain model, Mixture of User-adaptive Frequency FIlteriNg
(MUFFIN), operating through two complementary modules. (i) The global filtering
module (GFM) handles the entire frequency spectrum to capture comprehensive
behavioral patterns. (ii) The local filtering module (LFM) selectively
emphasizes important frequency bands without excluding information from other
ranges. (iii) In both modules, the user-adaptive filter (UAF) is adopted to
generate user-specific frequency filters tailored to individual unique
characteristics. Finally, by aggregating both modules, MUFFIN captures diverse
user behavioral patterns across the full frequency spectrum. Extensive
experiments show that MUFFIN consistently outperforms state-of-the-art
frequency-domain SR models over five benchmark datasets. The source code is
available at https://github.com/ilwoong100/MUFFIN.

### 3. [Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation](http://arxiv.org/pdf/2508.13745v1)

Authors: Shouxing Ma, Yawen Zeng, Shiqing Wu, Guandong Xu

Multi-modal recommender system focuses on utilizing rich modal information (
i.e., images and textual descriptions) of items to improve recommendation
performance. The current methods have achieved remarkable success with the
powerful structure modeling capability of graph neural networks. However, these
methods are often hindered by sparse data in real-world scenarios. Although
contrastive learning and homography ( i.e., homogeneous graphs) are employed to
address the data sparsity challenge, existing methods still suffer two main
limitations: 1) Simple multi-modal feature contrasts fail to produce effective
representations, causing noisy modal-shared features and loss of valuable
information in modal-unique features; 2) The lack of exploration of the
homograph relations between user interests and item co-occurrence results in
incomplete mining of user-item interplay.
  To address the above limitations, we propose a novel framework for
\textbf{R}\textbf{E}fining multi-mod\textbf{A}l cont\textbf{R}astive learning
and ho\textbf{M}ography relations (\textbf{REARM}). Specifically, we complement
multi-modal contrastive learning by employing meta-network and orthogonal
constraint strategies, which filter out noise in modal-shared features and
retain recommendation-relevant information in modal-unique features. To mine
homogeneous relationships effectively, we integrate a newly constructed user
interest graph and an item co-occurrence graph with the existing user
co-occurrence and item semantic graphs for graph learning. The extensive
experiments on three real-world datasets demonstrate the superiority of REARM
to various state-of-the-art baselines. Our visualization further shows an
improvement made by REARM in distinguishing between modal-shared and
modal-unique features. Code is available
\href{https://github.com/MrShouxingMa/REARM}{here}.

### 4. [Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate](http://arxiv.org/pdf/2508.13870v1)

Authors: Jiazheng Jing, Yinan Zhang, Chunyan Miao

The recent emergence of extreme climate events has significantly raised
awareness about sustainable living. In addition to developing energy-saving
materials and technologies, existing research mainly relies on traditional
methods that encourage behavioral shifts towards sustainability, which can be
overly demanding or only passively engaging. In this work, we propose to employ
recommendation systems to actively nudge users toward more sustainable choices.
We introduce Green Recommender Aligned with Personalized Eating (GRAPE), which
is designed to prioritize and recommend sustainable food options that align
with users' evolving preferences. We also design two innovative Green Loss
functions that cater to green indicators with either uniform or differentiated
priorities, thereby enhancing adaptability across a range of scenarios.
Extensive experiments on a real-world dataset demonstrate the effectiveness of
our GRAPE.

### 5. [CARE: Contextual Adaptation of Recommenders for LLM-based Conversational Recommendation](http://arxiv.org/pdf/2508.13889v1)

Authors: Chuang Li, Yang Deng, Hengchang Hu, See-Kiong Ng, Min-Yen Kan, Haizhou Li

We tackle the challenge of integrating large language models (LLMs) with
external recommender systems to enhance domain expertise in conversational
recommendation (CRS). Current LLM-based CRS approaches primarily rely on zero-
or few-shot methods for generating item recommendations based on user queries,
but this method faces two significant challenges: (1) without domain-specific
adaptation, LLMs frequently recommend items not in the target item space,
resulting in low recommendation accuracy; and (2) LLMs largely rely on dialogue
context for content-based recommendations, neglecting the collaborative
relationships among entities or item sequences. To address these limitations,
we introduce the CARE (Contextual Adaptation of Recommenders) framework. CARE
customizes LLMs for CRS tasks, and synergizes them with external recommendation
systems. CARE (a) integrates external recommender systems as domain experts,
producing recommendations through entity-level insights, and (b) enhances those
recommendations by leveraging contextual information for more accurate and
unbiased final recommendations using LLMs. Our results demonstrate that
incorporating external recommender systems with entity-level information
significantly enhances recommendation accuracy of LLM-based CRS by an average
of 54% and 25% for ReDial and INSPIRED datasets. The most effective strategy in
the CARE framework involves LLMs selecting and reranking candidate items that
external recommenders provide based on contextual insights. Our analysis
indicates that the CARE framework effectively addresses the identified
challenges and mitigates the popularity bias in the external recommender.

### 6. [Democratizing News Recommenders: Modeling Multiple Perspectives for News Candidate Generation with VQ-VAE](http://arxiv.org/pdf/2508.13978v1)

Authors: Hardy, Sebastian Padó, Amelie Wührl, Tanise Ceron

Current News Recommender Systems based on past clicks are designed for
engagement, but come at the cost of limiting diversity in the suggested
content. While diversity-aware algorithms exist, they suffer from two major
limitations. First, they fail to account for normative diversity, which
requires fair access to a broad range of perspectives. Second, they typically
apply diversity late in the system's pipeline, after a lot of content has
already been filtered out. Both limitations confine their effectiveness and
prevent them from promoting true normative diversity in news recommendations.
  We propose Aspect-Aware Candidate Generation (A2CG) to address these
limitations. Our framework introduces diversity into the earliest pipeline
stage and uses a configurable mechanism to align diversity with specific
democratic goals. A2CG represents each news article using multiple aspects of
perspectives (e.g., sentiment, political leaning, frame) and uses a Vector
Quantized Variational Autoencoder (VQ-VAE) to create a discrete, multi-faceted
representation. A decoder-only model then learns user preferences over these
aspect codes. We then inject diversity directly by reversing the sign on some
of the query vector's aspects during the candidate retrieval process, ensuring
a more diverse set of candidates.
  Our method, evaluated on the MIND dataset, enables a flexible trade-off
between personalization and diversity early in the recommendation pipeline. It
also generates more novel, diverse, and serendipitous candidates while
effectively taking into account aspects that strengthen democratic values.
These empirical results make it a promising approach for downstream
democratized news recommendation systems.

### 7. [AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System](http://arxiv.org/pdf/2508.13423v1)

Authors: Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu

In recent years, recommendation systems have evolved from providing a single
list of recommendations to offering a comprehensive suite of topic focused
services. To better accomplish this task, conversational recommendation systems
(CRS) have progressed from basic retrieval augmented LLM generation to agentic
systems with advanced reasoning and self correction capabilities. However,
agentic systems come with notable response latency, a longstanding challenge
for conversational recommendation systems. To balance the trade off between
handling complex queries and minimizing latency, we propose AdaptJobRec, the
first conversational job recommendation system that leverages autonomous agent
to integrate personalized recommendation algorithm tools. The system employs a
user query complexity identification mechanism to minimize response latency.
For straightforward queries, the agent directly selects the appropriate tool
for rapid responses. For complex queries, the agent uses the memory processing
module to filter chat history for relevant content, then passes the results to
the intelligent task decomposition planner, and finally executes the tasks
using personalized recommendation tools. Evaluation on Walmart's real world
career recommendation scenarios demonstrates that AdaptJobRec reduces average
response latency by up to 53.3% compared to competitive baselines, while
significantly improving recommendation accuracy.

### 8. [Understanding Distribution Structure on Calibrated Recommendation Systems](http://arxiv.org/pdf/2508.13568v1)

Authors: Diego Correa da Silva, Denis Robson Dantas Boaventura, Mayki dos Santos Oliveira, Eduardo Ferreira da Silva, Joel Machado Pires, Frederico Araújo Durão

Traditional recommender systems aim to generate a recommendation list
comprising the most relevant or similar items to the user's profile. These
approaches can create recommendation lists that omit item genres from the less
prominent areas of a user's profile, thereby undermining the user's experience.
To solve this problem, the calibrated recommendation system provides a
guarantee of including less representative areas in the recommended list. The
calibrated context works with three distributions. The first is from the user's
profile, the second is from the candidate items, and the last is from the
recommendation list. These distributions are G-dimensional, where G is the
total number of genres in the system. This high dimensionality requires a
different evaluation method, considering that traditional recommenders operate
in a one-dimensional data space. In this sense, we implement fifteen models
that help to understand how these distributions are structured. We evaluate the
users' patterns in three datasets from the movie domain. The results indicate
that the models of outlier detection provide a better understanding of the
structures. The calibrated system creates recommendation lists that act
similarly to traditional recommendation lists, allowing users to change their
groups of preferences to the same degree.

### 9. [UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion](http://arxiv.org/pdf/2508.13843v1)

Authors: Zihan Liang, Yufei Ma, ZhiPeng Qian, Huangyu Dai, Zihan Wang, Ben Chen, Chenyi Lei, Yuqing Ding, Han Li

Current e-commerce multimodal retrieval systems face two key limitations:
they optimize for specific tasks with fixed modality pairings, and lack
comprehensive benchmarks for evaluating unified retrieval approaches. To
address these challenges, we introduce UniECS, a unified multimodal e-commerce
search framework that handles all retrieval scenarios across image, text, and
their combinations. Our work makes three key contributions. First, we propose a
flexible architecture with a novel gated multimodal encoder that uses adaptive
fusion mechanisms. This encoder integrates different modality representations
while handling missing modalities. Second, we develop a comprehensive training
strategy to optimize learning. It combines cross-modal alignment loss (CMAL),
cohesive local alignment loss (CLAL), intra-modal contrastive loss (IMCL), and
adaptive loss weighting. Third, we create M-BEER, a carefully curated
multimodal benchmark containing 50K product pairs for e-commerce search
evaluation. Extensive experiments demonstrate that UniECS consistently
outperforms existing methods across four e-commerce benchmarks with fine-tuning
or zero-shot evaluation. On our M-BEER bench, UniECS achieves substantial
improvements in cross-modal tasks (up to 28\% gain in R@10 for text-to-image
retrieval) while maintaining parameter efficiency (0.2B parameters) compared to
larger models like GME-Qwen2VL (2B) and MM-Embed (8B). Furthermore, we deploy
UniECS in the e-commerce search platform of Kuaishou Inc. across two search
scenarios, achieving notable improvements in Click-Through Rate (+2.74\%) and
Revenue (+8.33\%). The comprehensive evaluation demonstrates the effectiveness
of our approach in both experimental and real-world settings. Corresponding
codes, models and datasets will be made publicly available at
https://github.com/qzp2018/UniECS.

### 10. [InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems](http://arxiv.org/pdf/2508.13930v1)

Authors: Matey Krastev, Miklos Hamar, Danilo Toapanta, Jesse Brouwers, Yibin Lei

This work revisits and extends synthetic query generation pipelines for
Neural Information Retrieval (NIR) by leveraging the InPars Toolkit, a
reproducible, end-to-end framework for generating training data using large
language models (LLMs). We first assess the reproducibility of the original
InPars, InPars-V2, and Promptagator pipelines on the SciFact benchmark and
validate their effectiveness using open-source reranker and generator models.
Building on this foundation, we introduce two key extensions to the pipeline:
(1) fine-tuning a query generator LLM via Contrastive Preference Optimization
(CPO) to improve the signal quality in generated queries, and (2) replacing
static prompt templates with dynamic, Chain-of-Thought (CoT) optimized prompts
using the DSPy framework. Our results show that both extensions reduce the need
for aggressive filtering while improving retrieval performance. All code,
models, and synthetic datasets are publicly released to support further
research at: \href{https://github.com/danilotpnta/IR2-project}{this https URL}.

### Machine Learning

### 1. [NovoMolGen: Rethinking Molecular Language Model Pretraining](http://arxiv.org/pdf/2508.13408v1)

Authors: Kamran Chitsaz, Roshan Balaji, Quentin Fournier, Nirav Pravinbhai Bhatt, Sarath Chandar

Designing de-novo molecules with desired property profiles requires efficient
exploration of the vast chemical space ranging from $10^{23}$ to $10^{60}$
possible synthesizable candidates. While various deep generative models have
been developed to design small molecules using diverse input representations,
Molecular Large Language Models (Mol-LLMs) based on string representations have
emerged as a scalable approach capable of exploring billions of molecules.
However, there remains limited understanding regarding how standard language
modeling practices such as textual representations, tokenization strategies,
model size, and dataset scale impact molecular generation performance. In this
work, we systematically investigate these critical aspects by introducing
NovoMolGen, a family of transformer-based foundation models pretrained on 1.5
billion molecules for de-novo molecule generation. Through extensive empirical
analyses, we identify a weak correlation between performance metrics measured
during pretraining and actual downstream performance, revealing important
distinctions between molecular and general NLP training dynamics. NovoMolGen
establishes new state-of-the-art results, substantially outperforming prior
Mol-LLMs and specialized generative models in both unconstrained and
goal-directed molecular generation tasks, thus providing a robust foundation
for advancing efficient and effective molecular modeling strategies.

### 2. [MAVIS: Multi-Objective Alignment via Value-Guided Inference-Time Search](http://arxiv.org/pdf/2508.13415v1)

Authors: Jeremy Carleton, Debajoy Mukherjee, Srinivas Shakkottai, Dileep Kalathil

Large Language Models (LLMs) are increasingly deployed across diverse
applications that demand balancing multiple, often conflicting, objectives --
such as helpfulness, harmlessness, or humor. Aligning outputs to user-specific
preferences in such multi-objective settings typically requires fine-tuning
models for each objective or preference configuration, which is computationally
expensive and inflexible. We introduce MAVIS -- Multi-Objective Alignment via
Value-Guided Inference-Time Search -- a lightweight inference-time alignment
framework that enables dynamic control over LLM behavior without modifying the
base model's weights. MAVIS trains a set of small value models, each
corresponding to a distinct objective. At inference time, these value models
are combined using user-specified weights to produce a tilting function that
adjusts the base model's output distribution toward desired trade-offs. The
value models are trained using a simple iterative algorithm that ensures
monotonic improvement of the KL-regularized policy. We show empirically that
MAVIS outperforms baselines that fine-tune per-objective models and combine
them post hoc, and even approaches the performance of the idealized setting
where models are fine-tuned for a user's exact preferences.

### 3. [ASAP: Unsupervised Post-training with Label Distribution Shift Adaptive Learning Rate](http://arxiv.org/pdf/2508.13445v1)

Authors: Heewon Park, Mugon Joe, Miru Kim, Minhae Kwon

In real-world applications, machine learning models face online label shift,
where label distributions change over time. Effective adaptation requires
careful learning rate selection: too low slows adaptation and too high causes
instability. We propose ASAP (Adaptive Shift Aware Post-training), which
dynamically adjusts the learning rate by computing the cosine distance between
current and previous unlabeled outputs and mapping it within a bounded range.
ASAP requires no labels, model ensembles, or past inputs, using only the
previous softmax output for fast, lightweight adaptation. Experiments across
multiple datasets and shift scenarios show ASAP consistently improves accuracy
and efficiency, making it practical for unsupervised model adaptation.

### 4. [Classifying Clinical Outcome of Epilepsy Patients with Ictal Chirp Embeddings](http://arxiv.org/pdf/2508.13476v1)

Authors: Nooshin Bahador, Milad Lankarany

This study presents a pipeline leveraging t-Distributed Stochastic Neighbor
Embedding (t-SNE) for interpretable visualizations of chirp features across
diverse outcome scenarios. The dataset, comprising chirp-based temporal,
spectral, and frequency metrics. Using t-SNE, local neighborhood relationships
were preserved while addressing the crowding problem through Student
t-distribution-based similarity optimization. Three classification tasks were
formulated on the 2D t-SNE embeddings: (1) distinguishing clinical success from
failure/no-resection, (2) separating high-difficulty from low-difficulty cases,
and (3) identifying optimal cases, defined as successful outcomes with minimal
clinical difficulty. Four classifiers, namely, Random Forests, Support Vector
Machines, Logistic Regression, and k-Nearest Neighbors, were trained and
evaluated using stratified 5-fold cross-validation. Across tasks, the Random
Forest and k-NN classifiers demonstrated superior performance, achieving up to
88.8% accuracy in optimal case detection (successful outcomes with minimal
clinical difficulty). Additionally, feature influence sensitivity maps were
generated using SHAP explanations applied to model predicting t-SNE
coordinates, revealing spatially localized feature importance within the
embedding space. These maps highlighted how specific chirp attributes drive
regional clustering and class separation, offering insights into the latent
structure of the data. The integrated framework showcases the potential of
interpretable embeddings and local feature attribution for clinical
stratification and decision support.

### 5. [Explainability of Algorithms](http://arxiv.org/pdf/2508.13529v1)

Authors: Andrés Páez

The opaqueness of many complex machine learning algorithms is often mentioned
as one of the main obstacles to the ethical development of artificial
intelligence (AI). But what does it mean for an algorithm to be opaque? Highly
complex algorithms such as artificial neural networks process enormous volumes
of data in parallel along multiple hidden layers of interconnected nodes,
rendering their inner workings epistemically inaccessible to any human being,
including their designers and developers; they are "black boxes" for all their
stakeholders. But opaqueness is not always the inevitable result of technical
complexity. Sometimes, the way an algorithm works is intentionally hidden from
view for proprietary reasons, especially in commercial automated decision
systems, creating an entirely different type of opaqueness. In the first part
of the chapter, we will examine these two ways of understanding opacity and the
ethical implications that stem from each of them. In the second part, we
explore the different explanatory methods that have been developed in computer
science to overcome an AI system's technical opaqueness. As the analysis shows,
explainable AI (XAI) still faces numerous challenges.

### 6. [CALYPSO: Forecasting and Analyzing MRSA Infection Patterns with Community and Healthcare Transmission Dynamics](http://arxiv.org/pdf/2508.13548v1)

Authors: Rituparna Datta, Jiaming Cui, Gregory R. Madden, Anil Vullikanti

Methicillin-resistant Staphylococcus aureus (MRSA) is a critical public
health threat within hospitals as well as long-term care facilities. Better
understanding of MRSA risks, evaluation of interventions and forecasting MRSA
rates are important public health problems. Existing forecasting models rely on
statistical or neural network approaches, which lack epidemiological
interpretability, and have limited performance. Mechanistic epidemic models are
difficult to calibrate and limited in incorporating diverse datasets. We
present CALYPSO, a hybrid framework that integrates neural networks with
mechanistic metapopulation models to capture the spread dynamics of infectious
diseases (i.e., MRSA) across healthcare and community settings. Our model
leverages patient-level insurance claims, commuting data, and healthcare
transfer patterns to learn region- and time-specific parameters governing MRSA
spread. This enables accurate, interpretable forecasts at multiple spatial
resolutions (county, healthcare facility, region, state) and supports
counterfactual analyses of infection control policies and outbreak risks. We
also show that CALYPSO improves statewide forecasting performance by over 4.5%
compared to machine learning baselines, while also identifying high-risk
regions and cost-effective strategies for allocating infection prevention
resources.

### 7. [Prediction of Hospital Associated Infections During Continuous Hospital Stays](http://arxiv.org/pdf/2508.13561v1)

Authors: Rituparna Datta, Methun Kamruzzaman, Eili Y. Klein, Gregory R Madden, Xinwei Deng, Anil Vullikanti, Parantapa Bhattacharya

The US Centers for Disease Control and Prevention (CDC), in 2019, designated
Methicillin-resistant Staphylococcus aureus (MRSA) as a serious antimicrobial
resistance threat. The risk of acquiring MRSA and suffering life-threatening
consequences due to it remains especially high for hospitalized patients due to
a unique combination of factors, including: co-morbid conditions, immuno
suppression, antibiotic use, and risk of contact with contaminated hospital
workers and equipment. In this paper, we present a novel generative
probabilistic model, GenHAI, for modeling sequences of MRSA test results
outcomes for patients during a single hospitalization. This model can be used
to answer many important questions from the perspectives of hospital
administrators for mitigating the risk of MRSA infections. Our model is based
on the probabilistic programming paradigm, and can be used to approximately
answer a variety of predictive, causal, and counterfactual questions. We
demonstrate the efficacy of our model by comparing it against discriminative
and generative machine learning models using two real-world datasets.

### 8. [A Generalized Learning Framework for Self-Supervised Contrastive Learning](http://arxiv.org/pdf/2508.13596v1)

Authors: Lingyu Si, Jingyao Wang, Wenwen Qiang

Self-supervised contrastive learning (SSCL) has recently demonstrated
superiority in multiple downstream tasks. In this paper, we generalize the
standard SSCL methods to a Generalized Learning Framework (GLF) consisting of
two parts: the aligning part and the constraining part. We analyze three
existing SSCL methods: BYOL, Barlow Twins, and SwAV, and show that they can be
unified under GLF with different choices of the constraining part. We further
propose empirical and theoretical analyses providing two insights into
designing the constraining part of GLF: intra-class compactness and inter-class
separability, which measure how well the feature space preserves the class
information of the inputs. However, since SSCL can not use labels, it is
challenging to design a constraining part that satisfies these properties. To
address this issue, we consider inducing intra-class compactness and
inter-class separability by iteratively capturing the dynamic relationship
between anchor and other samples and propose a plug-and-play method called
Adaptive Distribution Calibration (ADC) to ensure that samples that are near or
far from the anchor point in the original input space are closer or further
away from the anchor point in the feature space. Both the theoretical analysis
and the empirical evaluation demonstrate the superiority of ADC.

### 9. [Approximate Bayesian Inference via Bitstring Representations](http://arxiv.org/pdf/2508.13598v1)

Authors: Aleksanteri Sladek, Martin Trapp, Arno Solin

The machine learning community has recently put effort into quantized or
low-precision arithmetics to scale large models. This paper proposes performing
probabilistic inference in the quantized, discrete parameter space created by
these representations, effectively enabling us to learn a continuous
distribution using discrete parameters. We consider both 2D densities and
quantized neural networks, where we introduce a tractable learning approach
using probabilistic circuits. This method offers a scalable solution to manage
complex distributions and provides clear insights into model behavior. We
validate our approach with various models, demonstrating inference efficiency
without sacrificing accuracy. This work advances scalable, interpretable
machine learning by utilizing discrete approximations for probabilistic
computations.

### 10. [Text2Weight: Bridging Natural Language and Neural Network Weight Spaces](http://arxiv.org/pdf/2508.13633v1)

Authors: Bowen Tian, Wenshuo Chen, Zexi Li, Songning Lai, Jiemin Wu, Yutao Yue

How far are we really from automatically generating neural networks? While
neural network weight generation shows promise, current approaches struggle
with generalization to unseen tasks and practical application exploration. To
address this, we propose T2W, a diffusion transformer framework that generates
task-specific weights conditioned on natural language descriptions. T2W
hierarchically processes network parameters into uniform blocks, integrates
text embeddings from CLIP via a prior attention mechanism, and employs
adversarial training with weight-space augmentation to enhance generalization.
Experiments on Cifar100, Caltech256, and TinyImageNet demonstrate T2W's ability
to produce high-quality weights for unseen tasks, outperforming
optimization-based initialization and enabling novel applications such as
weight enhancement and text-guided model fusion. Our work bridges textual
semantics with weight-space dynamics, supported by an open-source dataset of
text-weight pairs, advancing the practicality of generative models in neural
network parameter synthesis. Our code is available on Github.

### Neural and Evolutionary Computing

### 1. [Zobrist Hash-based Duplicate Detection in Symbolic Regression](http://arxiv.org/pdf/2508.13859v1)

Authors: Bogdan Burlacu

Symbolic regression encompasses a family of search algorithms that aim to
discover the best fitting function for a set of data without requiring an a
priori specification of the model structure. The most successful and commonly
used technique for symbolic regression is Genetic Programming (GP), an
evolutionary search method that evolves a population of mathematical
expressions through the mechanism of natural selection. In this work we analyze
the efficiency of the evolutionary search in GP and show that many points in
the search space are re-visited and re-evaluated multiple times by the
algorithm, leading to wasted computational effort. We address this issue by
introducing a caching mechanism based on the Zobrist hash, a type of hashing
frequently used in abstract board games for the efficient construction and
subsequent update of transposition tables. We implement our caching approach
using the open-source framework Operon and demonstrate its performance on a
selection of real-world regression problems, where we observe up to 34\%
speedups without any detrimental effects on search quality. The hashing
approach represents a straightforward way to improve runtime performance while
also offering some interesting possibilities for adjusting search strategy
based on cached information.

### 2. [Multi-Plasticity Synergy with Adaptive Mechanism Assignment for Training Spiking Neural Networks](http://arxiv.org/pdf/2508.13673v1)

Authors: Yuzhe Liu, Xin Deng, Qiang Yu

Spiking Neural Networks (SNNs) are promising brain-inspired models known for
low power consumption and superior potential for temporal processing, but
identifying suitable learning mechanisms remains a challenge. Despite the
presence of multiple coexisting learning strategies in the brain, current SNN
training methods typically rely on a single form of synaptic plasticity, which
limits their adaptability and representational capability. In this paper, we
propose a biologically inspired training framework that incorporates multiple
synergistic plasticity mechanisms for more effective SNN training. Our method
enables diverse learning algorithms to cooperatively modulate the accumulation
of information, while allowing each mechanism to preserve its own relatively
independent update dynamics. We evaluated our approach on both static image and
dynamic neuromorphic datasets to demonstrate that our framework significantly
improves performance and robustness compared to conventional learning mechanism
models. This work provides a general and extensible foundation for developing
more powerful SNNs guided by multi-strategy brain-inspired learning.

### 3. [Encoding Optimization for Low-Complexity Spiking Neural Network Equalizers in IM/DD Systems](http://arxiv.org/pdf/2508.13783v1)

Authors: Eike-Manuel Edelmann, Alexander von Bank, Laurent Schmalen

Neural encoding parameters for spiking neural networks (SNNs) are typically
set heuristically. We propose a reinforcement learning-based algorithm to
optimize them. Applied to an SNN-based equalizer and demapper in an IM/DD
system, the method improves performance while reducing computational load and
network size.

### 4. [Timestep-Compressed Attack on Spiking Neural Networks through Timestep-Level Backpropagation](http://arxiv.org/pdf/2508.13812v1)

Authors: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Hyeongboo Baek, Brent ByungHoon Kang

State-of-the-art (SOTA) gradient-based adversarial attacks on spiking neural
networks (SNNs), which largely rely on extending FGSM and PGD frameworks, face
a critical limitation: substantial attack latency from multi-timestep
processing, rendering them infeasible for practical real-time applications.
This inefficiency stems from their design as direct extensions of ANN
paradigms, which fail to exploit key SNN properties. In this paper, we propose
the timestep-compressed attack (TCA), a novel framework that significantly
reduces attack latency. TCA introduces two components founded on key insights
into SNN behavior. First, timestep-level backpropagation (TLBP) is based on our
finding that global temporal information in backpropagation to generate
perturbations is not critical for an attack's success, enabling per-timestep
evaluation for early stopping. Second, adversarial membrane potential reuse
(A-MPR) is motivated by the observation that initial timesteps are
inefficiently spent accumulating membrane potential, a warm-up phase that can
be pre-calculated and reused. Our experiments on VGG-11 and ResNet-17 with the
CIFAR-10/100 and CIFAR10-DVS datasets show that TCA significantly reduces the
required attack latency by up to 56.6% and 57.1% compared to SOTA methods in
white-box and black-box settings, respectively, while maintaining a comparable
attack success rate.

### 5. [Dynamic Design of Machine Learning Pipelines via Metalearning](http://arxiv.org/pdf/2508.13436v1)

Authors: Edesio Alcobaça, André C. P. L. F. de Carvalho

Automated machine learning (AutoML) has democratized the design of machine
learning based systems, by automating model selection, hyperparameter tuning
and feature engineering. However, the high computational cost associated with
traditional search and optimization strategies, such as Random Search, Particle
Swarm Optimization and Bayesian Optimization, remains a significant challenge.
Moreover, AutoML systems typically explore a large search space, which can lead
to overfitting. This paper introduces a metalearning method for dynamically
designing search spaces for AutoML system. The proposed method uses historical
metaknowledge to select promising regions of the search space, accelerating the
optimization process. According to experiments conducted for this study, the
proposed method can reduce runtime by 89\% in Random Search and search space by
(1.8/13 preprocessor and 4.3/16 classifier), without compromising significant
predictive performance. Moreover, the proposed method showed competitive
performance when adapted to Auto-Sklearn, reducing its search space.
Furthermore, this study encompasses insights into meta-feature selection,
meta-model explainability, and the trade-offs inherent in search space
reduction strategies.

### Networking and Internet Architecture

### 1. [Fundamentals of Next-generation Network Planning](http://arxiv.org/pdf/2508.13469v1)

Authors: M. Umar Khan

The fifth-generation (5G) of cellular communications is expected to be
deployed in the next years to support a wide range of services with different
demands of peak data rates, latency and quality of experience (QoE). To support
higher data rates and latency requirements third-generation partnership project
(3GPP) has introduced numerology and bandwidth parts (BWPs), via new radio (NR)
for service-tailored resource allocation. Legacy 4G networks have generated
extensive data, which combined with crowd-sourced LTE infrastructure insights,
enables identification of high-traffic 5G deployment area (5GDA) for planning
new services. Given the mission-critical nature of 5G services, QoE is a big
challenge for MNOs to guarantee peak data rates for a defined percentage of
time. This work studies the fundamentals of 5G network planning methods that
reconciles coverage-capacity trade-offs through balanced radio network
dimensioning (RND), leveraging pragmatic NR modeling, and data-driven
strategies to minimize deployment costs and reduce cost-per-bit.

### 2. [Electromagnetic Signal Modulation Recognition based on Subgraph Embedding Learning](http://arxiv.org/pdf/2508.13474v1)

Authors: Bojun Zhang

Automatic Modulation Recognition (AMR) detects
  modulation schemes of received signals for further processing
  of signals without any priori information, which is critically
  important for civil spectrum regulation, information countermea sures, and
communication security. Due to the powerful feature
  extraction and classification capabilities of Deep Learning (DL),
  DL-based AMR algorithms have achieved excellent performance
  gains compared with traditional modulation detection algorithms.
  However, all existing DL-based AMR algorithms, to the best of
  our knowledge, are designed for specific channels and systems,
  because data dimension of the used training dataset is fixed. To
  this end, we takes the first step to propose a Subgraph Embedding
  Learning (SEL) structure to address the classical AMR problem,
  and the proposed algorithm is called SEL-AMR. Our algorithm
  treats the communication system as a subgraph and uses the
  relationship between samples to smooth the effects brought by
  noise and different channels to extract robust features. Thus,
  the proposed SEL-AMR algorithm can adapt to any dynamic
  channels and systems. We use 5 public real datasets and a small
  amount of simulation data to evaluate our SEL-AMR algorithm.
  Experimental results reveal that SEL-AMR can well adapt to
  different channels and systems, and always outperforms the state of-the-art
algorithms by improving up to 20% macro-average
  recognition precision and 30% recognition accuracy.

### 3. [CountingStars: Low-overhead Network-wide Measurement in LEO Mega-constellation Networks](http://arxiv.org/pdf/2508.13512v1)

Authors: Xiyuan Liu, Guano Liu, Xiucheng Tian, Wenting Wei

The high mobility of satellites in Low Earth Orbit (LEO) mega-constellations
induces a highly dynamic network topology, leading to many problems like
frequent service disruptions. To mitigate this, Packet-based Load Balancing
(PBLB) is employed. However, this paradigm shift introduces two critical
challenges for network measurement stemming from the requirement for port-level
granularity: memory inflation and severe hash collisions. To tackle these
challenges, we propose CountingStars, a low-overhead network-wide measurement
architecture. In the ground controller, CountingStars builds a digital twins
system to accurately predict the future network topology. This allows ground
controller to generate and distribute collision-free hash seeds to satellites
in advance. On the satellite, we introduce a port aggregation data structure
that decouples the unique flow identifier from its multi-port counter and
updates it through efficient bit operations, solving the memory inflation
caused by PBLB. Simulation results show that the memory usage of CountingStars
is reduced by 70\% on average, and the relative error of measurement is reduced
by 90\% on average. Implementation on FPGA shows its prospect to deploy in real
system.

### 4. [Architecture Considerations for ISAC in 6G](http://arxiv.org/pdf/2508.13736v1)

Authors: Sebastian Robitzsch, Laksh Bhatia, Konstantinos G. Filis, Neda Petreska, Michael Bahr, Pablo Picazo Martinez, Xi Li

ISAC is emerging as a foundational capability in 6G, enabling mobile networks
to not only offer communication services but also to sense and perceive their
environment at scale. This paper explores architectural considerations to
enable sensing in 6G, extending on recent developments by (pre-)standardisation
bodies such as 3GPP and ETSI. Selected ISAC use cases are presented from the
European MultiX project including associated potential functional system
requirements. The paper proposes a 6G system architecture that integrates newly
proposed NFs for the purpose of sensing and demonstrates how they are being
used in offering sensing as a service. Protocol stack adaptations for both
control and a newly proposed sensing plane are discussed.

### 5. [Security-as-a-Function for IDS/IPS in Softwarized Network and Applications to 5G Network Systems](http://arxiv.org/pdf/2508.13581v1)

Authors: Shivank Malik, Samaresh Bera

The service-based architecture of 5G network allows network operators to
place virtualized network functions on commodity hardware, unlike the
traditional vendor-specific hardware-based functionalities. However, it expands
the security vulnerabilities and threats to the 5G network. While there exist
several theoretical studies on network function placement and service routing,
a few focused on the security aspects of the 5G network systems.
  This paper focuses on safeguarding the 5G core network systems from DoS and
DDoS attacks by placing intrusion detection and prevention systems (IDS-IPS) as
virtualized network functions following the 5G standalone architecture. To
ensure the virtualized placement of IDS-IPS, first, we provide thorough virtual
machine (VM)-based and containerized implementation details and evaluate the
network performance with two scenarios, IDS and IPS, in the presence of TCP and
UDP applications. Second, we apply the VM-based implementation of IDS-IPS on a
softwarized 5G core network and study the network performances. The experiment
results on network throughput, latency, and packet drop reveal that the
softwarized IDS-IPS can meet the QoS requirements of 5G applications, while
safeguarding the network from DoS and DDoS attacks.

### 6. [Towards Timing Isolation for Mixed-Criticality Communication in Software-Defined Vehicles](http://arxiv.org/pdf/2508.13652v1)

Authors: Lóránt Meszlényi, Julius Kahle, Dominik Püllen, Stefan Kowalewski, Stefan Katzenbeisser, Alexandru Kampmann

As the automotive industry transitions toward centralized Linux-based
architectures, ensuring the predictable execution of mixed-criticality
applications becomes essential. However, concurrent use of the Linux network
stack introduces interference, resulting in unpredictable latency and jitter.
To address this challenge, we present a layered software architecture that
enforces timing isolation for Ethernet-based data exchange between
mixed-criticality applications on Linux-based automotive control units. Our
approach integrates traffic prioritization strategies at the middleware layer,
the network stack layer, and the hardware layer to achieve isolation across the
full software stack. At the middleware layer, we implement a fixed-priority,
non-preemptive scheduler to manage publishers of varying criticality. At the
network layer, we leverage the express data path (XDP) to route high-priority
data directly from the network interface driver into critical application
memory, bypassing the standard Linux network stack. At the hardware layer, we
dedicate a network interface card (NIC) queue exclusively to real-time traffic.
We demonstrate how our architecture performs in a Data Distribution Service
(DDS)-based system. Our evaluation shows that the approach leads to consistent
and predictable latencies for real-time traffic, even under heavy interference
from best-effort applications.

### 7. [BetaWeb: Towards a Blockchain-enabled Trustworthy Agentic Web](http://arxiv.org/pdf/2508.13787v1)

Authors: Zihan Guo, Yuanjian Zhou, Chenyi Wang, Linlin You, Minjie Bian, Weinan Zhang

The rapid development of large language models (LLMs) has significantly
propelled the development of artificial intelligence (AI) agents, which are
increasingly evolving into diverse autonomous entities, advancing the LLM-based
multi-agent systems (LaMAS). However, current agentic ecosystems remain
fragmented and closed. Establishing an interconnected and scalable paradigm for
Agentic AI has become a critical prerequisite. Although Agentic Web proposes an
open architecture to break the ecosystem barriers, its implementation still
faces core challenges such as privacy protection, data management, and value
measurement. Existing centralized or semi-centralized paradigms suffer from
inherent limitations, making them inadequate for supporting large-scale,
heterogeneous, and cross-domain autonomous interactions. To address these
challenges, this paper introduces the blockchain-enabled trustworthy Agentic
Web (BetaWeb). By leveraging the inherent strengths of blockchain, BetaWeb not
only offers a trustworthy and scalable infrastructure for LaMAS but also has
the potential to advance the Web paradigm from Web3 (centered on data
ownership) towards Web3.5, which emphasizes ownership of agent capabilities and
the monetization of intelligence. Beyond a systematic examination of the
BetaWeb framework, this paper presents a five-stage evolutionary roadmap,
outlining the path of LaMAS from passive execution to advanced collaboration
and autonomous governance. We also conduct a comparative analysis of existing
products and discuss key challenges of BetaWeb from multiple perspectives.
Ultimately, we argue that deep integration between blockchain and LaMAS can lay
the foundation for a resilient, trustworthy, and sustainably incentivized
digital ecosystem. A summary of the enabling technologies for each stage is
available at https://github.com/MatZaharia/BetaWeb.

### Robotics

### 1. [Switch4EAI: Leveraging Console Game Platform for Benchmarking Robotic Athletics](http://arxiv.org/pdf/2508.13444v1)

Authors: Tianyu Li, Jeonghwan Kim, Wontaek Kim, Donghoon Baek, Seungeun Rho, Sehoon Ha

Recent advances in whole-body robot control have enabled humanoid and legged
robots to execute increasingly agile and coordinated movements. However,
standardized benchmarks for evaluating robotic athletic performance in
real-world settings and in direct comparison to humans remain scarce. We
present Switch4EAI(Switch-for-Embodied-AI), a low-cost and easily deployable
pipeline that leverages motion-sensing console games to evaluate whole-body
robot control policies. Using Just Dance on the Nintendo Switch as a
representative example, our system captures, reconstructs, and retargets
in-game choreography for robotic execution. We validate the system on a Unitree
G1 humanoid with an open-source whole-body controller, establishing a
quantitative baseline for the robot's performance against a human player. In
the paper, we discuss these results, which demonstrate the feasibility of using
commercial games platform as physically grounded benchmarks and motivate future
work to for benchmarking embodied AI.

### 2. [CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models](http://arxiv.org/pdf/2508.13446v1)

Authors: Catherine Glossop, William Chen, Arjun Bhorkar, Dhruv Shah, Sergey Levine

Generalist robots should be able to understand and follow user instructions,
but current vision-language-action (VLA) models struggle with following
fine-grained commands despite providing a powerful architecture for mapping
open-vocabulary natural language instructions to robot actions. One cause for
this is a lack of semantic diversity and language grounding in existing robot
datasets and, specifically, a lack of fine-grained task diversity for similar
observations. To address this, we present a novel method to augment existing
robot datasets by leveraging vision language models to create counterfactual
labels. Our method improves the language-following capabilities of VLAs by
increasing the diversity and granularity of language grounding for robot
datasets by generating counterfactual language and actions. We evaluate the
resulting model's ability to follow language instructions, ranging from simple
object-centric commands to complex referential tasks, by conducting visual
language navigation experiments in 3 different indoor and outdoor environments.
Our experiments demonstrate that counterfactual relabeling, without any
additional data collection, significantly improves instruction-following in VLA
policies, making them competitive with state-of-the-art methods and increasing
success rate by 27% on navigation tasks.

### 3. [Unified Hierarchical MPC in Task Executing for Modular Manipulators across Diverse Morphologies](http://arxiv.org/pdf/2508.13513v1)

Authors: Maolin Lei, Edoardo Romiti, Arturo Laurenzi, Cheng Zhou, Wanli Xing, Liang Lu, Nikos G. Tsagarakis

This work proposes a unified Hierarchical Model Predictive Control (H-MPC)
for modular manipulators across various morphologies, as the controller can
adapt to different configurations to execute the given task without extensive
parameter tuning in the controller. The H-MPC divides the control process into
two levels: a high-level MPC and a low-level MPC. The high-level MPC predicts
future states and provides trajectory information, while the low-level MPC
refines control actions by updating the predictive model based on this
high-level information. This hierarchical structure allows for the integration
of kinematic constraints and ensures smooth joint-space trajectories, even near
singular configurations. Moreover, the low-level MPC incorporates secondary
linearization by leveraging predictive information from the high-level MPC,
effectively capturing the second-order Taylor expansion information of the
kinematic model while still maintaining a linearized model formulation. This
approach not only preserves the simplicity of a linear control model but also
enhances the accuracy of the kinematic representation, thereby improving
overall control precision and reliability. To validate the effectiveness of the
control policy, we conduct extensive evaluations across different manipulator
morphologies and demonstrate the execution of pick-and-place tasks in
real-world scenarios.

### 4. [A Three-Level Whole-Body Disturbance Rejection Control Framework for Dynamic Motions in Legged Robots](http://arxiv.org/pdf/2508.13531v1)

Authors: Bolin Li, Gewei Zuo, Zhixiang Wang, Xiaotian Ke, Lijun Zhu, Han Ding

This paper presents a control framework designed to enhance the stability and
robustness of legged robots in the presence of uncertainties, including model
uncertainties, external disturbances, and faults. The framework enables the
full-state feedback estimator to estimate and compensate for uncertainties in
whole-body dynamics of the legged robots. First, we propose a novel moving
horizon extended state observer (MH-ESO) to estimate uncertainties and mitigate
noise in legged systems, which can be integrated into the framework for
disturbance compensation. Second, we introduce a three-level whole-body
disturbance rejection control framework (T-WB-DRC). Unlike the previous
two-level approach, this three-level framework considers both the plan based on
whole-body dynamics without uncertainties and the plan based on dynamics with
uncertainties, significantly improving payload transportation, external
disturbance rejection, and fault tolerance. Third, simulations of both humanoid
and quadruped robots in the Gazebo simulator demonstrate the effectiveness and
versatility of T-WB-DRC. Finally, extensive experimental trials on a quadruped
robot validate the robustness and stability of the system when using T-WB-DRC
under various disturbance conditions.

### 5. [Assessing Pedestrian Behavior Around Autonomous Cleaning Robots in Public Spaces: Findings from a Field Observation](http://arxiv.org/pdf/2508.13699v1)

Authors: Maren Raab, Linda Miller, Zhe Zeng, Pascal Jansen, Martin Baumann, Johannes Kraus

As autonomous robots become more common in public spaces, spontaneous
encounters with laypersons are more frequent. For this, robots need to be
equipped with communication strategies that enhance momentary transparency and
reduce the probability of critical situations. Adapting these robotic
strategies requires consideration of robot movements, environmental conditions,
and user characteristics and states. While numerous studies have investigated
the impact of distraction on pedestrians' movement behavior, limited research
has examined this behavior in the presence of autonomous robots. This research
addresses the impact of robot type and robot movement pattern on distracted and
undistracted pedestrians' movement behavior. In a field setting, unaware
pedestrians were videotaped while moving past two working, autonomous cleaning
robots. Out of N=498 observed pedestrians, approximately 8% were distracted by
smartphones. Distracted and undistracted pedestrians did not exhibit
significant differences in their movement behaviors around the robots. Instead,
both the larger sweeping robot and the offset rectangular movement pattern
significantly increased the number of lateral adaptations compared to the
smaller cleaning robot and the circular movement pattern. The offset
rectangular movement pattern also led to significantly more close lateral
adaptations. Depending on the robot type, the movement patterns led to
differences in the distances of lateral adaptations. The study provides initial
insights into pedestrian movement behavior around an autonomous cleaning robot
in public spaces, contributing to the growing field of HRI research.

### 6. [Blast Hole Seeking and Dipping -- The Navigation and Perception Framework in a Mine Site Inspection Robot](http://arxiv.org/pdf/2508.13785v1)

Authors: Liyang Liu, Ehsan Mihankhah, Nathan Wallace, Javier Martinez, Andrew J. Hill

In open-pit mining, holes are drilled into the surface of the excavation site
and detonated with explosives to facilitate digging. These blast holes need to
be inspected internally for investigation of downhole material types and
properties. Knowing these properties can lead to significant savings in
material handling costs in downstream processes. Manual hole inspection is slow
and expensive, with major limitations in revealing the geometric and geological
properties of the holes and their contents. This has been the motivation for
the development of our autonomous mine-site inspection robot - "DIPPeR". In
this paper, the automation aspect of the project is explained. We present a
robust blast hole seeking and detection framework that enables target-based
navigation and accurate down-hole sensor positioning. The pipeline first
processes point-cloud data collected by the on-board LiDAR sensors, extracting
the cone-shaped volume of drill-waste above the ground. By projecting the 3D
cone points into a virtual depth image, segmentation is achieved in the 2D
domain, yielding a circular hole at the image centre and a collared cone face.
We then identify the hole centre using a robust detection module while
suppressing non-maximum candidates, ensuring precise sensor placement for
down-hole inspection and avoiding collisions with the cavity wall. To enable
autonomous hole-seeking, the pipeline automatically adjusts its projection
parameters during robot navigation to account for variations in point sparsity
and hole opening size, ensuring a consistent hole appearance in 2D images. This
allows continuous tracking of the target hole as the robot approaches the goal
point. We demonstrate the effectiveness of our navigation and perception system
in both high-fidelity simulation environments and on-site field tests. A
demonstration video is available at
"https://www.youtube.com/watch?v=fRNbcBcaSqE".

### 7. [Trajectory Tracking and Stabilization of Quadrotors Using Deep Koopman Model Predictive Control](http://arxiv.org/pdf/2508.13795v1)

Authors: Haitham El-Hussieny

This paper presents a data-driven control framework for quadrotor systems
that integrates a deep Koopman operator with model predictive control (DK-MPC).
The deep Koopman operator is trained on sampled flight data to construct a
high-dimensional latent representation in which the nonlinear quadrotor
dynamics are approximated by linear models. This linearization enables the
application of MPC to efficiently optimize control actions over a finite
prediction horizon, ensuring accurate trajectory tracking and stabilization.
The proposed DK-MPC approach is validated through a series of
trajectory-following and point-stabilization numerical experiments, where it
demonstrates superior tracking accuracy and significantly lower computation
time compared to conventional nonlinear MPC. These results highlight the
potential of Koopman-based learning methods to handle complex quadrotor
dynamics while meeting the real-time requirements of embedded flight control.
Future work will focus on extending the framework to more agile flight
scenarios and improving robustness against external disturbances.

### 8. [Driving Style Recognition Like an Expert Using Semantic Privileged Information from Large Language Models](http://arxiv.org/pdf/2508.13881v1)

Authors: Zhaokun Chen, Chaopeng Zhang, Xiaohan Li, Wenshuo Wang, Gentiane Venture, Junqiang Xi

Existing driving style recognition systems largely depend on low-level
sensor-derived features for training, neglecting the rich semantic reasoning
capability inherent to human experts. This discrepancy results in a fundamental
misalignment between algorithmic classifications and expert judgments. To
bridge this gap, we propose a novel framework that integrates Semantic
Privileged Information (SPI) derived from large language models (LLMs) to align
recognition outcomes with human-interpretable reasoning. First, we introduce
DriBehavGPT, an interactive LLM-based module that generates natural-language
descriptions of driving behaviors. These descriptions are then encoded into
machine learning-compatible representations via text embedding and
dimensionality reduction. Finally, we incorporate them as privileged
information into Support Vector Machine Plus (SVM+) for training, enabling the
model to approximate human-like interpretation patterns. Experiments across
diverse real-world driving scenarios demonstrate that our SPI-enhanced
framework outperforms conventional methods, achieving F1-score improvements of
7.6% (car-following) and 7.9% (lane-changing). Importantly, SPI is exclusively
used during training, while inference relies solely on sensor data, ensuring
computational efficiency without sacrificing performance. These results
highlight the pivotal role of semantic behavioral representations in improving
recognition accuracy while advancing interpretable, human-centric driving
systems.

### 9. [Toward an Interaction-Centered Approach to Robot Trustworthiness](http://arxiv.org/pdf/2508.13976v1)

Authors: Carlo Mazzola, Hassan Ali, Kristína Malinovská, Igor Farkaš

As robots get more integrated into human environments, fostering
trustworthiness in embodied robotic agents becomes paramount for an effective
and safe human-robot interaction (HRI). To achieve that, HRI applications must
promote human trust that aligns with robot skills and avoid misplaced trust or
overtrust, which can pose safety risks and ethical concerns. To achieve that,
HRI applications must promote human trust that aligns with robot skills and
avoid misplaced trust or overtrust, which can pose safety risks and ethical
concerns. In this position paper, we outline an interaction-based framework for
building trust through mutual understanding between humans and robots. We
emphasize two main pillars: human awareness and transparency, referring to the
robot ability to interpret human actions accurately and to clearly communicate
its intentions and goals, respectively. By integrating these two pillars,
robots can behave in a manner that aligns with human expectations and needs
while providing their human partners with both comprehension and control over
their actions. We also introduce four components that we think are important
for bridging the gap between a human-perceived sense of trust and a robot true
capabilities.

### 10. [Train Once, Deploy Anywhere: Realize Data-Efficient Dynamic Object Manipulation](http://arxiv.org/pdf/2508.14042v1)

Authors: Zhuoling Li, Xiaoyang Wu, Zhenhua Xu, Hengshuang Zhao

Realizing generalizable dynamic object manipulation is important for
enhancing manufacturing efficiency, as it eliminates specialized engineering
for various scenarios. To this end, imitation learning emerges as a promising
paradigm, leveraging expert demonstrations to teach a policy manipulation
skills. Although the generalization of an imitation learning policy can be
improved by increasing demonstrations, demonstration collection is
labor-intensive. To address this problem, this paper investigates whether
strong generalization in dynamic object manipulation is achievable with only a
few demonstrations. Specifically, we develop an entropy-based theoretical
framework to quantify the optimization of imitation learning. Based on this
framework, we propose a system named Generalizable Entropy-based Manipulation
(GEM). Extensive experiments in simulated and real tasks demonstrate that GEM
can generalize across diverse environment backgrounds, robot embodiments,
motion dynamics, and object geometries. Notably, GEM has been deployed in a
real canteen for tableware collection. Without any in-scene demonstration, it
achieves a success rate of over 97% across more than 10,000 operations.

### Software Engineering

### 1. [The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget](http://arxiv.org/pdf/2508.13666v1)

Authors: Dangfeng Pan, Zhensu Sun, Cenyuan Zhang, David Lo, Xiaoning Du

Source code is usually formatted with elements like indentation and newlines
to improve readability for human developers. However, these visual aids do not
seem to be beneficial for large language models (LLMs) in the same way since
the code is processed as a linear sequence of tokens. Furthermore, these
additional tokens can lead to increased computational costs and longer response
times for LLMs. If such formatting elements are non-essential to LLMs, we can
reduce such costs by removing them from the code. To figure out the role played
by formatting elements, we conduct a comprehensive empirical study to evaluate
the impact of code formatting on LLM performance and efficiency. Through
large-scale experiments on Fill-in-the-Middle Code Completion tasks across four
programming languages (Java, Python, C++, C\#) and ten LLMs-including both
commercial and open-source models-we systematically analyze token count and
performance when formatting elements are removed. Key findings indicate that
LLMs can maintain performance across formatted code and unformatted code,
achieving an average input token reduction of 24.5\% with negligible output
token reductions. This makes code format removal a practical optimization
strategy for improving LLM efficiency. Further exploration reveals that both
prompting and fine-tuning LLMs can lead to significant reductions (up to
36.1\%) in output code length without compromising correctness. To facilitate
practical applications, we develop a bidirectional code transformation tool for
format processing, which can be seamlessly integrated into existing LLM
inference workflows, ensuring both human readability and LLM efficiency.

### 2. [Structural and Connectivity Patterns in the Maven Central Software Dependency Network](http://arxiv.org/pdf/2508.13819v1)

Authors: Daniel Ogenrwot, John Businge, Shaikh Arifuzzaman

Understanding the structural characteristics and connectivity patterns of
large-scale software ecosystems is critical for enhancing software reuse,
improving ecosystem resilience, and mitigating security risks. In this paper,
we investigate the Maven Central ecosystem, one of the largest repositories of
Java libraries, by applying network science techniques to its dependency graph.
Leveraging the Goblin framework, we extracted a sample consisting of the top
5,000 highly connected artifacts based on their degree centrality and then
performed breadth-first search (BFS) expansion from each selected artifact as a
seed node, traversing the graph outward to capture all libraries and releases
reachable those seed nodes. This sampling strategy captured the immediate
structural context surrounding these libraries resulted in a curated graph
comprising of 1.3 million nodes and 20.9 million edges. We conducted a
comprehensive analysis of this graph, computing degree distributions,
betweenness centrality, PageRank centrality, and connected components
graph-theoretic metrics. Our results reveal that Maven Central exhibits a
highly interconnected, scale-free, and small-world topology, characterized by a
small number of infrastructural hubs that support the majority of projects.
Further analysis using PageRank and betweenness centrality shows that these
hubs predominantly consist of core ecosystem infrastructure, including testing
frameworks and general-purpose utility libraries. While these hubs facilitate
efficient software reuse and integration, they also pose systemic risks;
failures or vulnerabilities affecting these critical nodes can have widespread
and cascading impacts throughout the ecosystem.

### 3. [Tight Inter-Core Cache Contention Analysis for WCET Estimation on Multicore Systems](http://arxiv.org/pdf/2508.13863v1)

Authors: Shuai Zhao, Jieyu Jiang, Shenlin Cai, Yaowei Liang, Chen Jie, Yinjie Fang, Wei Zhang, Guoquan Zhang, Yaoyao Gu, Xiang Xiao, Wei Qin, Xiangzhen Ouyang, Wanli Chang

WCET (Worst-Case Execution Time) estimation on multicore architecture is
particularly challenging mainly due to the complex accesses over cache shared
by multiple cores. Existing analysis identifies possible contentions between
parallel tasks by leveraging the partial order of the tasks or their program
regions. Unfortunately, they overestimate the number of cache misses caused by
a remote block access without considering the actual cache state and the number
of accesses. This paper reports a new analysis for inter-core cache contention.
Based on the order of program regions in a task, we first identify memory
references that could be affected if a remote access occurs in a region.
Afterwards, a fine-grained contention analysis is constructed that computes the
number of cache misses based on the access quantity of local and remote blocks.
We demonstrate that the overall inter-core cache interference of a task can be
obtained via dynamic programming. Experiments show that compared to existing
methods, the proposed analysis reduces inter-core cache interference and WCET
estimations by 52.31% and 8.94% on average, without significantly increasing
computation overhead.

### 4. [Large Language Models as Visualization Agents for Immersive Binary Reverse Engineering](http://arxiv.org/pdf/2508.13413v1)

Authors: Dennis Brown, Samuel Mulder

Immersive virtual reality (VR) offers affordances that may reduce cognitive
complexity in binary reverse engineering (RE), enabling embodied and external
cognition to augment the RE process through enhancing memory, hypothesis
testing, and visual organization. In prior work, we applied a cognitive systems
engineering approach to identify an initial set of affordances and implemented
a VR environment to support RE through spatial persistence and interactivity.
In this work, we extend that platform with an integrated large language model
(LLM) agent capable of querying binary analysis tools, answering technical
questions, and dynamically generating immersive 3D visualizations in alignment
with analyst tasks. We describe the system architecture and our evaluation
process and results. Our pilot study shows that while LLMs can generate
meaningful 3D call graphs (for small programs) that align with design
principles, output quality varies widely. This work raises open questions about
the potential for LLMs to function as visualization agents, constructing 3D
representations that reflect cognitive design principles without explicit
training.

### 5. [Conflicting Scores, Confusing Signals: An Empirical Study of Vulnerability Scoring Systems](http://arxiv.org/pdf/2508.13644v1)

Authors: Viktoria Koscinski, Mark Nelson, Ahmet Okutan, Robert Falso, Mehdi Mirakhorli

Accurately assessing software vulnerabilities is essential for effective
prioritization and remediation. While various scoring systems exist to support
this task, their differing goals, methodologies and outputs often lead to
inconsistent prioritization decisions. This work provides the first
large-scale, outcome-linked empirical comparison of four publicly available
vulnerability scoring systems: the Common Vulnerability Scoring System (CVSS),
the Stakeholder-Specific Vulnerability Categorization (SSVC), the Exploit
Prediction Scoring System (EPSS), and the Exploitability Index. We use a
dataset of 600 real-world vulnerabilities derived from four months of
Microsoft's Patch Tuesday disclosures to investigate the relationships between
these scores, evaluate how they support vulnerability management task, how
these scores categorize vulnerabilities across triage tiers, and assess their
ability to capture the real-world exploitation risk. Our findings reveal
significant disparities in how scoring systems rank the same vulnerabilities,
with implications for organizations relying on these metrics to make
data-driven, risk-based decisions. We provide insights into the alignment and
divergence of these systems, highlighting the need for more transparent and
consistent exploitability, risk, and severity assessments.

### 6. [COMPASS: A Multi-Dimensional Benchmark for Evaluating Code Generation in Large Language Models](http://arxiv.org/pdf/2508.13757v1)

Authors: James Meaden, Michał Jarosz, Piotr Jodłowski, Grigori Melnik

Current code generation benchmarks focus primarily on functional correctness
while overlooking two critical aspects of real-world programming: algorithmic
efficiency and code quality. We introduce COMPASS (COdility's Multi-dimensional
Programming ASSessment), a comprehensive evaluation framework that assesses
code generation across three dimensions: correctness, efficiency, and quality.
COMPASS consists of 50 competitive programming problems from real Codility
competitions, providing authentic human baselines from 393,150 submissions.
Unlike existing benchmarks that treat algorithmically inefficient solutions
identically to optimal ones provided they pass test cases, COMPASS
systematically evaluates runtime efficiency and code quality using
industry-standard analysis tools. Our evaluation of three leading
reasoning-enhanced models, Anthropic Claude Opus 4, Google Gemini 2.5 Pro, and
OpenAI O4-Mini-High, reveals that models achieving high correctness scores do
not necessarily produce efficient algorithms or maintainable code. These
findings highlight the importance of evaluating more than just correctness to
truly understand the real-world capabilities of code generation models. COMPASS
serves as a guiding framework, charting a path for future research toward AI
systems that are robust, reliable, and ready for production use.

### 7. [Agentic DraCor and the Art of Docstring Engineering: Evaluating MCP-empowered LLM Usage of the DraCor API](http://arxiv.org/pdf/2508.13774v1)

Authors: Peer Trilcke, Ingo Börner, Henny Sluyter-Gäthje, Daniil Skorinkin, Frank Fischer, Carsten Milling

This paper reports on the implementation and evaluation of a Model Context
Protocol (MCP) server for DraCor, enabling Large Language Models (LLM) to
autonomously interact with the DraCor API. We conducted experiments focusing on
tool selection and application by the LLM, employing a qualitative approach
that includes systematic observation of prompts to understand how LLMs behave
when using MCP tools, evaluating "Tool Correctness", "Tool-Calling Efficiency",
and "Tool-Use Reliability". Our findings highlight the importance of "Docstring
Engineering", defined as reflexively crafting tool documentation to optimize
LLM-tool interaction. Our experiments demonstrate both the promise of agentic
AI for research in Computational Literary Studies and the essential
infrastructure development needs for reliable Digital Humanities
infrastructures.

### 8. [Reactive Semantics for User Interface Description Languages](http://arxiv.org/pdf/2508.13610v1)

Authors: Basile Pesin, Celia Picard, Cyril Allignol

User Interface Description Languages (UIDLs) are high-level languages that
facilitate the development of Human-Machine Interfaces, such as Graphical User
Interface (GUI) applications. They usually provide first-class primitives to
specify how the program reacts to an external event (user input, network
message), and how data flows through the program. Although these
domain-specific languages are now widely used to implement safety-critical
GUIs, little work has been invested in their formalization and verification.
  In this paper, we propose a denotational semantic model for a core reactive
UIDL, Smalite, which we argue is expressive enough to encode constructs from
more realistic languages. This preliminary work may be used as a stepping stone
to produce a formally verified compiler for UIDLs.

### Social and Information Networks

### 1. [Towards a general diffusion-based information quality assessment model](http://arxiv.org/pdf/2508.13927v1)

Authors: Anthony Lopes Temporao, Mickael Temporao, Corentin Vande Kerckhove, Flavio Abreu Araujo

The rapid and unregulated dissemination of information in the digital era has
amplified the global "infodemic," complicating the identification of high
quality information. We present a lightweight, interpretable and non-invasive
framework for assessing information quality based solely on diffusion dynamics,
demonstrated here in the context of academic publications. Using a
heterogeneous dataset of 29,264 sciences, technology, engineering, mathematics
(STEM) and social science papers from ArnetMiner and OpenAlex, we model the
diffusion network of each paper as a set of three theoretically motivated
features: diversity, timeliness, and salience. A Generalized Additive Model
(GAM) trained on these features achieved Pearson correlations of 0.8468 for
next-year citation gain and up to 97.8% accuracy in predicting high-impact
papers. Feature relevance studies reveal timeliness and salience as the most
robust predictors, while diversity offers less stable benefits in the academic
setting but may be more informative in social media contexts. The framework's
transparency, domain-agnostic design, and minimal feature requirements position
it as a scalable tool for global information quality assessment, opening new
avenues for moving beyond binary credibility labels toward richer,
diffusion-informed evaluation metrics.

### 2. [Heterogeneous Influence Maximization in User Recommendation](http://arxiv.org/pdf/2508.13517v1)

Authors: Hongru Hou, Jiachen Sun, Wenqing Lin, Wendong Bi, Xiangrong Wang, Deqing Yang

User recommendation systems enhance user engagement by encouraging users to
act as inviters to interact with other users (invitees), potentially fostering
information propagation. Conventional recommendation methods typically focus on
modeling interaction willingness. Influence-Maximization (IM) methods focus on
identifying a set of users to maximize the information propagation. However,
existing methods face two significant challenges. First, recommendation methods
fail to unleash the candidates' spread capability. Second, IM methods fail to
account for the willingness to interact. To solve these issues, we propose two
models named HeteroIR and HeteroIM. HeteroIR provides an intuitive solution to
unleash the dissemination potential of user recommendation systems. HeteroIM
fills the gap between the IM method and the recommendation task, improving
interaction willingness and maximizing spread coverage. The HeteroIR introduces
a two-stage framework to estimate the spread profits. The HeteroIM
incrementally selects the most influential invitee to recommend and rerank
based on the number of reverse reachable (RR) sets containing inviters and
invitees. RR set denotes a set of nodes that can reach a target via
propagation. Extensive experiments show that HeteroIR and HeteroIM
significantly outperform the state-of-the-art baselines with the p-value <
0.05. Furthermore, we have deployed HeteroIR and HeteroIM in Tencent's online
gaming platforms and gained an 8.5\% and 10\% improvement in the online A/B
test, respectively. Implementation codes are available at
https://github.com/socialalgo/HIM.

### 3. [Exit Stories: Using Reddit Self-Disclosures to Understand Disengagement from Problematic Communities](http://arxiv.org/pdf/2508.13837v1)

Authors: Shruti Phadke

Online platforms like Reddit are increasingly becoming popular for
individuals sharing personal experiences of leaving behind social, ideological,
and political groups. Specifically, a series of "ex-" subreddits on Reddit
allow users to recount their departures from commitments such as religious
affiliations, manosphere communities, conspiracy theories or political beliefs,
and lifestyle choices. Understanding the natural process through which users
exit, especially from problematic groups such as conspiracy theory communities
and the manosphere, can provide valuable insights for designing interventions
targeting disengagement from harmful ideologies. This paper presents an
in-depth exploration of 15K exit stories across 131 subreddits, focusing on
five key areas: religion, manosphere, conspiracy theories, politics, and
lifestyle. Using a transdisciplinary framework that incorporates theories from
social psychology, organizational behavior, and violent extremism studies, this
work identifies a range of factors contributing to disengagement. The results
describe how disengagement from problematic groups, such as conspiracy theories
and the manosphere, is a multi-faceted process that is qualitatively different
than disengaging from more established social structures, such as religions or
political ideologies. This research further highlights the need for moving
beyond interventions that treat conspiracy theorizing solely as an information
problem and contributes insights for future research focusing on offering
mental health interventions and support in exit communities.

### 4. [Trust and Reputation in Data Sharing: A Survey](http://arxiv.org/pdf/2508.14028v1)

Authors: Wenbo Wu, George Konstantinidis

Data sharing is the fuel of the galloping artificial intelligence economy,
providing diverse datasets for training robust models. Trust between data
providers and data consumers is widely considered one of the most important
factors for enabling data sharing initiatives. Concerns about data sensitivity,
privacy breaches, and misuse contribute to reluctance in sharing data across
various domains. In recent years, there has been a rise in technological and
algorithmic solutions to measure, capture and manage trust, trustworthiness,
and reputation in what we collectively refer to as Trust and Reputation
Management Systems (TRMSs). Such approaches have been developed and applied to
different domains of computer science, such as autonomous vehicles, or IoT
networks, but there have not been dedicated approaches to data sharing and its
unique characteristics. In this survey, we examine TRMSs from a data-sharing
perspective, analyzing how they assess the trustworthiness of both data and
entities across different environments. We develop novel taxonomies for system
designs, trust evaluation framework, and evaluation metrics for both data and
entity, and we systematically analyze the applicability of existing TRMSs in
data sharing. Finally, we identify open challenges and propose future research
directions to enhance the explainability, comprehensiveness, and accuracy of
TRMSs in large-scale data-sharing ecosystems.

### Systems and Control

### 1. [Power-Series Approach to Moment-Matching-Based Model Reduction of MIMO Polynomial Nonlinear Systems](http://arxiv.org/pdf/2508.13595v1)

Authors: Chao Huang, Alessandro Astolfi

The model reduction problem for high-order multi-input, multi-output (MIMO)
polynomial nonlinear systems based on moment matching is addressed. The
technique of power-series decomposition is exploited: this decomposes the
solution of the nonlinear PDE characterizing the center manifold into the
solutions of a series of recursively defined Sylvester equations. This approach
allows yielding nonlinear reduced-order models in very much the same way as in
the linear case (e.g. analytically). Algorithms are proposed for obtaining the
order and the parameters of the reduced-order models with precision of degree
$\kappa$. The approach also provides new insights into the nonlinear moment
matching problem: first, a lower bound for the order of the reduced-order model
is obtained, which, in the MIMO case, can be strictly less than the number of
matched moments; second, it is revealed that the lower bound is affected by the
ratio of the number of the input and output channels; third, it is shown that
under mild conditions, a nonlinear reduced-order model can always be
constructed with either a linear state equation or a linear output equation.

### 2. [Scalable Sensor Placement for Cyclic Networks with Observability Guarantees: Application to Water Distribution Networks](http://arxiv.org/pdf/2508.13604v1)

Authors: J. J. H. van Gemert, V. Breschi, D. R. Yntema, K. J. Keesman, M. Lazar

Optimal sensor placement is essential for state estimation and effective
network monitoring. As known in the literature, this problem becomes
particularly challenging in large-scale undirected or bidirected cyclic
networks with parametric uncertainties, such as water distribution networks
(WDNs), where pipe resistance and demand patterns are often unknown. Motivated
by the challenges of cycles, parametric uncertainties, and scalability, this
paper proposes a sensor placement algorithm that guarantees structural
observability for cyclic and acyclic networks with parametric uncertainties. By
leveraging a graph-based strategy, the proposed method efficiently addresses
the computational complexities of large-scale networks. To demonstrate the
algorithm's effectiveness, we apply it to several EPANET benchmark WDNs. Most
notably, the developed algorithm solves the sensor placement problem with
guaranteed structured observability for the L-town WDN with 1694 nodes and 124
cycles in under 0.1 seconds.

### 3. [Transient Stability Analysis for Grid Following Converters in Low-Inertia Power Systems by Direct Method](http://arxiv.org/pdf/2508.13641v1)

Authors: Fangyuan Sun, Ruisheng Diao, Ruiyuan Zeng, Zhanning Liu, Baorong Zhou, Junjie Li, Wangqianyun Tang

With the increased penetration of renewable energy and reduced proportion of
synchronous generators, the low-inertia characteristics of todays power system
become prominent, and the transient stability issue of grid following converter
(GFLC) under low inertia system (LIS) condition becomes critical. There are two
prominent problems in the transient stability analysis of GFLC-LIS. The angular
dynamic of LIS increases the complexity of transient stability analysis, and
the nonlinear, possibly negative damping of GFLC makes it difficult to
guarantee the conservative of the traditional methods. These problems make the
traditional methods inapplicable. In this paper, the transient stability
analysis of GFLC LIS is investigated to provide an accurate estimation of the
attraction boundary and critical clearance time (CCT). Firstly, a dynamic model
of GFLC-LIS is constructed, considering the phase-locked loop (PLL)-based GFLC
dynamics and swing equation-based LIS dynamics. The frequency mutation of PLL
at fault occurrence and clearing time is also considered. Secondly, a Zubov
based transient stability analysis method is proposed, which can construct the
energy function in a way that is different from the traditional conservation of
energy perspective and can address the negative damping issue. Moreover, the
accuracy of the CCT estimation is analyzed, and the influences of LIS
parameters on transient stability are illustrated. Finally, simulation
experiments are carried out to verify the effectiveness of the proposed method

### 4. [Singularity-free prescribed performance guaranteed control for perturbed system](http://arxiv.org/pdf/2508.13726v1)

Authors: Yiwei Liu

This paper addresses the prescribed performance control (PPC) challenge for
high-order nonlinear systems affected by mismatched disturbances. The research
aims to prevent singularity issues arising from error boundary violations
during abrupt changes in reference trajectories. We introduce a novel
transformation function with infinite-order differentiability at connection
points, advancing beyond mere continuous differentiability. Utilizing this
transformation function, we develop a comprehensive transformation strategy
that ensures: (1) errors remain within prescribed boundaries when reference
trajectories are smooth, and (2) errors return to prescribed boundaries within
a specified timeframe following abrupt changes in reference trajectories.
Additionally, the complexity explosion issue inherent in backstepping design is
effectively resolved. Simulation results corroborate the validity of the
proposed theoretical advancements.

### 5. [Energy Management and Wake-up for IoT Networks Powered by Energy Harvesting](http://arxiv.org/pdf/2508.13825v1)

Authors: David Ernesto Ruiz-Guirola, Samuel Montejo-Sanchez, Israel Leyva-Mayorga, Zhu Han, Petar Popovski, Onel L. A. Lopez

The rapid growth of the Internet of Things (IoT) presents sustainability
challenges such as increased maintenance requirements and overall higher energy
consumption. This motivates self-sustainable IoT ecosystems based on Energy
Harvesting (EH). This paper treats IoT deployments in which IoT devices (IoTDs)
rely solely on EH to sense and transmit information about events/alarms to a
base station (BS). The objective is to effectively manage the duty cycling of
the IoTDs to prolong battery life and maximize the relevant data sent to the
BS. The BS can also wake up specific IoTDs if extra information about an event
is needed upon initial detection. We propose a K-nearest neighbors (KNN)-based
duty cycling management to optimize energy efficiency and detection accuracy by
considering spatial correlations among IoTDs' activity and their EH process. We
evaluate machine learning approaches, including reinforcement learning (RL) and
decision transformers (DT), to maximize information captured from events while
managing energy consumption. Significant improvements over the state-ofthe-art
approaches are obtained in terms of energy saving by all three proposals, KNN,
RL, and DT. Moreover, the RL-based solution approaches the performance of a
genie-aided benchmark as the number of IoTDs increases.

### 6. [LLMind 2.0: Distributed IoT Automation with Natural Language M2M Communication and Lightweight LLM Agents](http://arxiv.org/pdf/2508.13920v1)

Authors: Yuyang Du, Qun Yang, Liujianfu Wang, Jingqi Lin, Hongwei Cui, Soung Chang Liew

Recent advances in large language models (LLMs) have sparked interest in
their application to IoT and automation systems, particularly for facilitating
device management through natural language instructions. However, existing
centralized approaches face significant scalability challenges when managing
and coordinating the collaboration between IoT devices of diverse capabilities
in large-scale heterogeneous IoT systems. This paper introduces LLMind 2.0, a
distributed IoT automation framework that addresses the scalability challenges
through lightweight LLM-empowered device agents via natural language-based
machine-to-machine (M2M) communication. Unlike previous LLM-controlled
automation systems that rely on a centralized coordinator to generate
device-specific code to be executed on individual devices, LLMind 2.0
distributes intelligence across individual devices through lightweight LLMs
embedded in IoT devices. The central coordinator translates human instructions
into simple subtasks described in natural human language, which are then
processed by device-specific agents to generate device-specific code locally at
the associated devices. This approach transcends device heterogeneity barriers
by using natural language as a unified communication medium, enabling seamless
collaboration between devices from different manufacturers. The system
incorporates several key innovations: a Retrieval-Augmented Generation (RAG)
mechanism for accurate subtask-to-API mapping, fine-tuned lightweight LLMs for
reliable code generation, and a finite state machine-based task execution
framework. Experimental validation in multi-robot warehouse scenarios and
real-world WiFi network deployments demonstrates significant improvements in
scalability, reliability, and privacy protection compared to the centralized
approach.

### 7. [Modeling and Control of AWOISV: A Filtered Tube-Based MPC Approach for Simultaneous Tracking of Lateral Position and Heading Angle](http://arxiv.org/pdf/2508.13457v1)

Authors: Xu Yang, Jun Ni, Hengyang Feng, Feiyu Wang, Tiezhen Wang

An all-wheel omni-directional independent steering vehicle (AWOISV) is a
specialized all-wheel independent steering vehicle with each wheel capable of
steering up to 90{\deg}, enabling unique maneuvers like yaw and diagonal
movement. This paper introduces a theoretical steering radius angle and
sideslip angle (\( \theta_R \)-\(\beta_R \)) representation, based on the
position of the instantaneous center of rotation relative to the wheel rotation
center, defining the motion modes and switching criteria for AWOISVs. A
generalized \( v\)-\(\beta\)-\(r \) dynamic model is developed with forward
velocity \(v\), sideslip angle \(\beta\), and yaw rate \(r\) as states, and
\(\theta_R\) and \(\beta_R\) as control inputs. This model decouples
longitudinal and lateral motions into forward and rotational motions, allowing
seamless transitions across all motion modes under specific conditions. A
filtered tube-based linear time-varying MPC (FT-LTVMPC) strategy is proposed,
achieving simultaneous tracking of lateral position and arbitrary heading
angles, with robustness to model inaccuracies and parameter uncertainties.
Co-simulation and hardware-in-loop (HIL) experiments confirm that FT-LTVMPC
enables high-precision control of both position and heading while ensuring
excellent real-time performance.

### 8. [System-Level Performance and Communication Tradeoff in Networked Control with Predictions](http://arxiv.org/pdf/2508.13475v1)

Authors: Yifei Wu, Jing Yu, Tongxin Li

Distributed control of large-scale systems is challenging due to the need for
scalable and localized communication and computation. In this work, we
introduce a Predictive System-Level Synthesis PredSLS framework that designs
controllers by jointly integrating communication constraints and local
disturbance predictions into an affine feedback structure. Rather than focusing
on the worst-case uncertainty, PredSLS leverages both current state feedback
and future system disturbance predictions to achieve distributed control of
networked systems. In particular, PredSLS enables a unified system synthesis of
the optimal $\kappa$-localized controller, therefore outperforms approaches
with post hoc communication truncation, as was commonly seen in the literature.
The PredSLS framework can be naturally decomposed into spatial and temporal
components for efficient and parallelizable computation across the network,
yielding a regret upper bound that explicitly depends on the prediction error
and communication range. Our regret analysis not only reveals a non-monotonic
trade-off between control performance and communication range when prediction
errors are present, but also guides the identification of an optimal size for
local communication neighborhoods, thereby enabling the co-design of controller
and its underlying communication topology.

### 9. [MuFlex: A Scalable, Physics-based Platform for Multi-Building Flexibility Analysis and Coordination](http://arxiv.org/pdf/2508.13532v1)

Authors: Ziyan Wu, Ivan Korolija, Rui Tang

With the increasing penetration of renewable generation on the power grid,
maintaining system balance requires coordinated demand flexibility from
aggregations of buildings. Reinforcement learning (RL) has been widely explored
for building controls because of its model-free nature. Open-source simulation
testbeds are essential not only for training RL agents but also for fairly
benchmarking control strategies. However, most building-sector testbeds target
single buildings; multi-building platforms are relatively limited and typically
rely on simplified models (e.g., Resistance-Capacitance) or data-driven
approaches, which lack the ability to fully capture the physical intricacies
and intermediate variables necessary for interpreting control performance.
Moreover, these platforms often impose fixed inputs, outputs, and model
formats, restricting their applicability as benchmarking tools across diverse
control scenarios. To address these gaps, MuFlex, a scalable, open-source
platform for benchmarking and testing control strategies for multi-building
flexibility coordination, was developed in this study. MuFlex enables
synchronous information exchange across EnergyPlus building models and adheres
to the latest OpenAI Gym interface, providing a modular, standardized RL
implementation. The platform capabilities were demonstrated in a case study
coordinating demand flexibility across four office buildings using the Soft
Actor-Critic algorithm with carefully fine-tuned hyperparameters. The results
show that aggregating the four buildings flexibility reduced total peak demand
below a specified threshold while maintaining indoor environmental quality.

### 10. [Repeater Swarm-Assisted Cellular Systems: Interaction Stability and Performance Analysis](http://arxiv.org/pdf/2508.13593v1)

Authors: Jianan Bai, Anubhab Chowdhury, Anders Hansson, Erik G. Larsson

We consider a cellular massive MIMO system where swarms of wireless repeaters
are deployed to improve coverage. These repeaters are full-duplex relays with
small form factors that receive and instantaneously retransmit signals. They
can be deployed in a plug-and-play manner at low cost, while being transparent
to the network--conceptually they are active channel scatterers with
amplification capabilities. Two fundamental questions need to be addressed in
repeater deployments: (I) How can we prevent destructive effects of positive
feedback caused by inter-repeater interaction (i.e., each repeater receives and
amplifies signals from others)? (ii) How much performance improvement can be
achieved given that repeaters also inject noise and may introduce more
interference? To answer these questions, we first derive a generalized Nyquist
stability criterion for the repeater swarm system, and provide an easy-to-check
stability condition. Then, we study the uplink performance and develop an
efficient iterative algorithm that jointly optimizes the repeater gains, user
transmit powers, and receive combining weights to maximize the weighted sum
rate while ensuring system stability. Numerical results corroborate our
theoretical findings and show that the repeaters can significantly improve the
system performance, both in sub-6 GHz and millimeter-wave bands. The results
also warrant careful deployment to fully realize the benefits of repeaters, for
example, by ensuring a high probability of line-of-sight links between
repeaters and the base station.

### Machine Learning (Statistics Category)

### 1. [Disentangled Deep Smoothed Bootstrap for Fair Imbalanced Regression](http://arxiv.org/pdf/2508.13829v1)

Authors: Samuel Stocksieker, Denys pommeret, Arthur Charpentier

Imbalanced distribution learning is a common and significant challenge in
predictive modeling, often reducing the performance of standard algorithms.
Although various approaches address this issue, most are tailored to
classification problems, with a limited focus on regression. This paper
introduces a novel method to improve learning on tabular data within the
Imbalanced Regression (IR) framework, which is a critical problem. We propose
using Variational Autoencoders (VAEs) to model and define a latent
representation of data distributions. However, VAEs can be inefficient with
imbalanced data like other standard approaches. To address this, we develop an
innovative data generation method that combines a disentangled VAE with a
Smoothed Bootstrap applied in the latent space. We evaluate the efficiency of
this method through numerical comparisons with competitors on benchmark
datasets for IR.

### 2. [Smooth Flow Matching](http://arxiv.org/pdf/2508.13831v1)

Authors: Jianbin Tan, Anru R. Zhang

Functional data, i.e., smooth random functions observed over a continuous
domain, are increasingly available in areas such as biomedical research, health
informatics, and epidemiology. However, effective statistical analysis for
functional data is often hindered by challenges such as privacy constraints,
sparse and irregular sampling, infinite dimensionality, and non-Gaussian
structures. To address these challenges, we introduce a novel framework named
Smooth Flow Matching (SFM), tailored for generative modeling of functional data
to enable statistical analysis without exposing sensitive real data. Built upon
flow-matching ideas, SFM constructs a semiparametric copula flow to generate
infinite-dimensional functional data, free from Gaussianity or low-rank
assumptions. It is computationally efficient, handles irregular observations,
and guarantees the smoothness of the generated functions, offering a practical
and flexible solution in scenarios where existing deep generative methods are
not applicable. Through extensive simulation studies, we demonstrate the
advantages of SFM in terms of both synthetic data quality and computational
efficiency. We then apply SFM to generate clinical trajectory data from the
MIMIC-IV patient electronic health records (EHR) longitudinal database. Our
analysis showcases the ability of SFM to produce high-quality surrogate data
for downstream statistical tasks, highlighting its potential to boost the
utility of EHR data for clinical applications.

### 3. [Online Conformal Selection with Accept-to-Reject Changes](http://arxiv.org/pdf/2508.13838v1)

Authors: Kangdao Liu, Huajun Xi, Chi-Man Vong, Hongxin Wei

Selecting a subset of promising candidates from a large pool is crucial
across various scientific and real-world applications. Conformal selection
offers a distribution-free and model-agnostic framework for candidate selection
with uncertainty quantification. While effective in offline settings, its
application to online scenarios, where data arrives sequentially, poses
challenges. Notably, conformal selection permits the deselection of previously
selected candidates, which is incompatible with applications requiring
irreversible selection decisions. This limitation is particularly evident in
resource-intensive sequential processes, such as drug discovery, where
advancing a compound to subsequent stages renders reversal impractical. To
address this issue, we extend conformal selection to an online Accept-to-Reject
Changes (ARC) procedure: non-selected data points can be reconsidered for
selection later, and once a candidate is selected, the decision is
irreversible. Specifically, we propose a novel conformal selection method,
Online Conformal Selection with Accept-to-Reject Changes (dubbed OCS-ARC),
which incorporates online Benjamini-Hochberg procedure into the candidate
selection process. We provide theoretical guarantees that OCS-ARC controls the
false discovery rate (FDR) at or below the nominal level at any timestep under
both i.i.d. and exchangeable data assumptions. Additionally, we theoretically
show that our approach naturally extends to multivariate response settings.
Extensive experiments on synthetic and real-world datasets demonstrate that
OCS-ARC significantly improves selection power over the baseline while
maintaining valid FDR control across all examined timesteps.

### 4. [Diffusion-Driven High-Dimensional Variable Selection](http://arxiv.org/pdf/2508.13890v1)

Authors: Minjie Wang, Xiaotong Shen, Wei Pan

Variable selection for high-dimensional, highly correlated data has long been
a challenging problem, often yielding unstable and unreliable models. We
propose a resample-aggregate framework that exploits diffusion models' ability
to generate high-fidelity synthetic data. Specifically, we draw multiple
pseudo-data sets from a diffusion model fitted to the original data, apply any
off-the-shelf selector (e.g., lasso or SCAD), and store the resulting inclusion
indicators and coefficients. Aggregating across replicas produces a stable
subset of predictors with calibrated stability scores for variable selection.
Theoretically, we show that the proposed method is selection consistent under
mild assumptions. Because the generative model imports knowledge from large
pre-trained weights, the procedure naturally benefits from transfer learning,
boosting power when the observed sample is small or noisy. We also extend the
framework of aggregating synthetic data to other model selection problems,
including graphical model selection, and statistical inference that supports
valid confidence intervals and hypothesis tests. Extensive simulations show
consistent gains over the lasso, stability selection, and knockoff baselines,
especially when predictors are strongly correlated, achieving higher
true-positive rates and lower false-discovery proportions. By coupling
diffusion-based data augmentation with principled aggregation, our method
advances variable selection methodology and broadens the toolkit for
interpretable, statistically rigorous analysis in complex scientific
applications.

### 5. [Generalisation and benign over-fitting for linear regression onto random functional covariates](http://arxiv.org/pdf/2508.13895v1)

Authors: Andrew Jones, Nick Whiteley

We study theoretical predictive performance of ridge and ridge-less
least-squares regression when covariate vectors arise from evaluating $p$
random, means-square continuous functions over a latent metric space at $n$
random and unobserved locations, subject to additive noise. This leads us away
from the standard assumption of i.i.d. data to a setting in which the $n$
covariate vectors are exchangeable but not independent in general. Under an
assumption of independence across dimensions, $4$-th order moment, and other
regularity conditions, we obtain probabilistic bounds on a notion of predictive
excess risk adapted to our random functional covariate setting, making use of
recent results of Barzilai and Shamir. We derive convergence rates in regimes
where $p$ grows suitably fast relative to $n$, illustrating interplay between
ingredients of the model in determining convergence behaviour and the role of
additive covariate noise in benign-overfitting.

### 6. [Minimizing the Weighted Number of Tardy Jobs: Data-Driven Heuristic for Single-Machine Scheduling](http://arxiv.org/pdf/2508.13703v1)

Authors: Nikolai Antonov, Prěmysl Šůcha, Mikoláš Janota, Jan Hůla

Existing research on single-machine scheduling is largely focused on exact
algorithms, which perform well on typical instances but can significantly
deteriorate on certain regions of the problem space. In contrast, data-driven
approaches provide strong and scalable performance when tailored to the
structure of specific datasets. Leveraging this idea, we focus on a
single-machine scheduling problem where each job is defined by its weight,
duration, due date, and deadline, aiming to minimize the total weight of tardy
jobs. We introduce a novel data-driven scheduling heuristic that combines
machine learning with problem-specific characteristics, ensuring feasible
solutions, which is a common challenge for ML-based algorithms. Experimental
results demonstrate that our approach significantly outperforms the
state-of-the-art in terms of optimality gap, number of optimal solutions, and
adaptability across varied data scenarios, highlighting its flexibility for
practical applications. In addition, we conduct a systematic exploration of ML
models, addressing a common gap in similar studies by offering a detailed model
selection process and providing insights into why the chosen model is the best
fit.

### 7. [Multi-User Contextual Cascading Bandits for Personalized Recommendation](http://arxiv.org/pdf/2508.13981v1)

Authors: Jiho Park, Huiwen Jia

We introduce a Multi-User Contextual Cascading Bandit model, a new
combinatorial bandit framework that captures realistic online advertising
scenarios where multiple users interact with sequentially displayed items
simultaneously. Unlike classical contextual bandits, MCCB integrates three key
structural elements: (i) cascading feedback based on sequential arm exposure,
(ii) parallel context sessions enabling selective exploration, and (iii)
heterogeneous arm-level rewards. We first propose Upper Confidence Bound with
Backward Planning (UCBBP), a UCB-style algorithm tailored to this setting, and
prove that it achieves a regret bound of $\widetilde{O}(\sqrt{THN})$ over $T$
episodes, $H$ session steps, and $N$ contexts per episode. Motivated by the
fact that many users interact with the system simultaneously, we introduce a
second algorithm, termed Active Upper Confidence Bound with Backward Planning
(AUCBBP), which shows a strict efficiency improvement in context scaling, i.e.,
user scaling, with a regret bound of $\widetilde{O}(\sqrt{T+HN})$. We validate
our theoretical findings via numerical experiments, demonstrating the empirical
effectiveness of both algorithms under various settings.

### 8. [Uncertainty-Aware PCA for Arbitrarily Distributed Data Modeled by Gaussian Mixture Models](http://arxiv.org/pdf/2508.13990v1)

Authors: Daniel Klötzl, Ozan Tastekin, David Hägele, Marina Evers, Daniel Weiskopf

Multidimensional data is often associated with uncertainties that are not
well-described by normal distributions. In this work, we describe how such
distributions can be projected to a low-dimensional space using
uncertainty-aware principal component analysis (UAPCA). We propose to model
multidimensional distributions using Gaussian mixture models (GMMs) and derive
the projection from a general formulation that allows projecting arbitrary
probability density functions. The low-dimensional projections of the densities
exhibit more details about the distributions and represent them more faithfully
compared to UAPCA mappings. Further, we support including user-defined weights
between the different distributions, which allows for varying the importance of
the multidimensional distributions. We evaluate our approach by comparing the
distributions in low-dimensional space obtained by our method and UAPCA to
those obtained by sample-based projections.

### 9. [A PC Algorithm for Max-Linear Bayesian Networks](http://arxiv.org/pdf/2508.13967v1)

Authors: Carlos Améndola, Benjamin Hollering, Francesco Nowell

Max-linear Bayesian networks (MLBNs) are a relatively recent class of
structural equation models which arise when the random variables involved have
heavy-tailed distributions. Unlike most directed graphical models, MLBNs are
typically not faithful to d-separation and thus classical causal discovery
algorithms such as the PC algorithm or greedy equivalence search can not be
used to accurately recover the true graph structure. In this paper, we begin
the study of constraint-based discovery algorithms for MLBNs given an oracle
for testing conditional independence in the true, unknown graph. We show that
if the oracle is given by the $\ast$-separation criteria in the true graph,
then the PC algorithm remains consistent despite the presence of additional CI
statements implied by $\ast$-separation. We also introduce a new causal
discovery algorithm named "PCstar" which assumes faithfulness to
$C^\ast$-separation and is able to orient additional edges which cannot be
oriented with only d- or $\ast$-separation.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-20 PST.

### 1. [What counts as plagiarism? AI-generated papers pose new risks](https://www.nature.com/articles/d41586-025-02616-5)

Authors: Ananya

### 2. [Automatic detection of cognitive events using machine learning and understanding models’ interpretations of human cognition](https://www.nature.com/articles/s41598-025-16165-4)

Authors: Quang Dang et al.

### 3. [Machine learning-enhanced fully coupled fluid–solid interaction models for proppant dynamics in hydraulic fractures](https://www.nature.com/articles/s41598-025-15837-5)

Authors: Dennis Delali Kwesi Wayo et al.

### 4. [A scalable machine learning strategy for resource allocation in database](https://www.nature.com/articles/s41598-025-14962-5)

Authors: Fady Nashat Manhary et al.

### 5. [Intelligent resource allocation in internet of things using random forest and clustering techniques](https://www.nature.com/articles/s41598-025-15931-8)

Authors: Nahideh Derakhshanfard et al.

### 6. [Multi-axis compression fusion network for vehicle re-identification](https://www.nature.com/articles/s41598-025-15854-4)

Authors: Tengda Ma et al.

### 7. [Electron flow matching for generative reaction mechanism prediction](https://www.nature.com/articles/s41586-025-09426-9)

Authors: Joonyoung F. Joung et al.

### 8. [Visual language transformer framework for multimodal dance performance evaluation and progression monitoring](https://www.nature.com/articles/s41598-025-16345-2)

Authors: Lei Chen

### 9. [Prediction of antibiotic resistance from antibiotic susceptibility testing results from surveillance data using machine learning](https://www.nature.com/articles/s41598-025-14078-w)

Authors: Swetha Valavarasu et al.

### 10. [Neural radiance fields assisted by image features for UAV scene reconstruction](https://www.nature.com/articles/s41598-025-16386-7)

Authors: Zhihong Chen et al.

