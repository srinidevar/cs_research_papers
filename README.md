# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-10 17:00:26.150992 PST.

### Artificial Intelligence

### 1. [SCC-recursiveness in infinite argumentation (extended version)](http://arxiv.org/pdf/2507.06852v1)

Authors: Uri Andrews, Luca San Mauro

Argumentation frameworks (AFs) are a foundational tool in artificial
intelligence for modeling structured reasoning and conflict. SCC-recursiveness
is a well-known design principle in which the evaluation of arguments is
decomposed according to the strongly connected components (SCCs) of the attack
graph, proceeding recursively from "higher" to "lower" components. While
SCC-recursive semantics such as \cft and \stgt have proven effective for finite
AFs, Baumann and Spanring showed the failure of SCC-recursive semantics to
generalize reliably to infinite AFs due to issues with well-foundedness.
  We propose two approaches to extending SCC-recursiveness to the infinite
setting. We systematically evaluate these semantics using Baroni and Giacomin's
established criteria, showing in particular that directionality fails in
general. We then examine these semantics' behavior in finitary frameworks,
where we find some of our semantics satisfy directionality. These results
advance the theory of infinite argumentation and lay the groundwork for
reasoning systems capable of handling unbounded or evolving domains.

### 2. [First Return, Entropy-Eliciting Explore](http://arxiv.org/pdf/2507.07017v1)

Authors: Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, Qian Liu, Ge Zhang, Zejun Ma

Reinforcement Learning from Verifiable Rewards (RLVR) improves the reasoning
abilities of Large Language Models (LLMs) but it struggles with unstable
exploration. We propose FR3E (First Return, Entropy-Eliciting Explore), a
structured exploration framework that identifies high-uncertainty decision
points in reasoning trajectories and performs targeted rollouts to construct
semantically grounded intermediate feedback. Our method provides targeted
guidance without relying on dense supervision. Empirical results on
mathematical reasoning benchmarks(AIME24) show that FR3E promotes more stable
training, produces longer and more coherent responses, and increases the
proportion of fully correct trajectories. These results highlight the
framework's effectiveness in improving LLM reasoning through more robust and
structured exploration.

### 3. [EA: An Event Autoencoder for High-Speed Vision Sensing](http://arxiv.org/pdf/2507.06459v1)

Authors: Riadul Islam, Joey Mulé, Dhandeep Challagundla, Shahmir Rizvi, Sean Carson

High-speed vision sensing is essential for real-time perception in
applications such as robotics, autonomous vehicles, and industrial automation.
Traditional frame-based vision systems suffer from motion blur, high latency,
and redundant data processing, limiting their performance in dynamic
environments. Event cameras, which capture asynchronous brightness changes at
the pixel level, offer a promising alternative but pose challenges in object
detection due to sparse and noisy event streams. To address this, we propose an
event autoencoder architecture that efficiently compresses and reconstructs
event data while preserving critical spatial and temporal features. The
proposed model employs convolutional encoding and incorporates adaptive
threshold selection and a lightweight classifier to enhance recognition
accuracy while reducing computational complexity. Experimental results on the
existing Smart Event Face Dataset (SEFD) demonstrate that our approach achieves
comparable accuracy to the YOLO-v4 model while utilizing up to $35.5\times$
fewer parameters. Implementations on embedded platforms, including Raspberry Pi
4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8
FPS. The proposed classifier exhibits up to 87.84x better FPS than the
state-of-the-art and significantly improves event-based vision performance,
making it ideal for low-power, high-speed applications in real-time edge
computing.

### 4. [SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam](http://arxiv.org/pdf/2507.06464v1)

Authors: Hanyang Peng, Shuang Qin, Yue Yu, Fangqing Jiang, Hui Wang, Wen Gao

Adam has proven remarkable successful in training deep neural networks, but
the mechanisms underlying its empirical successes and limitations remain
underexplored. In this study, we demonstrate that the effectiveness of Adam
stems largely from its similarity to SignSGD in robustly handling large
gradient fluctuations, yet it is also vulnerable to destabilizing loss spikes
due to its uncontrolled update scaling. To enhance the advantage of Adam and
mitigate its limitation, we propose SignSoftSGD (S3), a novel optimizer with
three key innovations. \emph{First}, S3 generalizes the sign-like update by
employing a flexible $p$-th order momentum ($p \geq 1$) in the denominator,
departing from the conventional second-order momentum (variance)
preconditioning. This design enables enhanced performance while achieving
stable training even with aggressive learning rates. \emph{Second}, S3
minimizes the occurrences of loss spikes through unified exponential moving
average coefficients for numerator and denominator momenta, which inherently
bound updates to $[-1, 1]$ and simplify hyperparameter tuning. \emph{Third}, S3
incorporates an equivalent Nesterov's accelerated gradient(NAG) module,
accelerating convergence without memory overhead. Theoretically, we prove that
S3 achieves the optimal convergence rate of
$O\left(\frac{1}{T^{\sfrac{1}{4}}}\right)$ for general nonconvex stochastic
optimization under weak assumptions. Extensive experiments across a range of
vision and language tasks show that \textsf{\small S3} not only converges more
rapidly and improves performance but also rarely experiences loss spikes, even
with a \textbf{$\bm{10 \times}$} larger learning rate. In fact, S3 delivers
performance comparable to or better than AdamW with \textbf{$2 \times$} the
training steps, establishing its efficacy in both efficiency and final task
performance.

### 5. [Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models](http://arxiv.org/pdf/2507.06466v1)

Authors: Aaron Dharna, Cong Lu, Jeff Clune

Multi-agent interactions have long fueled innovation, from natural
predator-prey dynamics to the space race. Self-play (SP) algorithms try to
harness these dynamics by pitting agents against ever-improving opponents,
thereby creating an implicit curriculum toward learning high-quality solutions.
However, SP often fails to produce diverse solutions and can get stuck in
locally optimal behaviors. We introduce Foundation-Model Self-Play (FMSP), a
new direction that leverages the code-generation capabilities and vast
knowledge of foundation models (FMs) to overcome these challenges by leaping
across local optima in policy space. We propose a family of approaches: (1)
\textbf{Vanilla Foundation-Model Self-Play (vFMSP)} continually refines agent
policies via competitive self-play; (2) \textbf{Novelty-Search Self-Play
(NSSP)} builds a diverse population of strategies, ignoring performance; and
(3) the most promising variant, \textbf{Quality-Diveristy Self-Play (QDSP)},
creates a diverse set of high-quality policies by combining the diversity of
NSSP and refinement of vFMSP. We evaluate FMSPs in Car Tag, a
continuous-control pursuer-evader setting, and in Gandalf, a simple AI safety
simulation in which an attacker tries to jailbreak an LLM's defenses. In Car
Tag, FMSPs explore a wide variety of reinforcement learning, tree search, and
heuristic-based methods, to name just a few. In terms of discovered policy
quality, \ouralgo and vFMSP surpass strong human-designed strategies. In
Gandalf, FMSPs can successfully automatically red-team an LLM, breaking through
and jailbreaking six different, progressively stronger levels of defense.
Furthermore, FMSPs can automatically proceed to patch the discovered
vulnerabilities. Overall, FMSPs represent a promising new research frontier of
improving self-play with foundation models, opening fresh paths toward more
creative and open-ended strategy discovery

### 6. [MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models](http://arxiv.org/pdf/2507.06502v1)

Authors: Yiwen Liu, Chenyu Zhang, Junjie Song, Siqi Chen, Sun Yin, Zihan Wang, Lingming Zeng, Yuji Cao, Junming Jiao

As a prominent data modality task, time series forecasting plays a pivotal
role in diverse applications. With the remarkable advancements in Large
Language Models (LLMs), the adoption of LLMs as the foundational architecture
for time series modeling has gained significant attention. Although existing
models achieve some success, they rarely both model time and frequency
characteristics in a pretraining-finetuning paradigm leading to suboptimal
performance in predictions of complex time series, which requires both modeling
periodicity and prior pattern knowledge of signals. We propose MoFE-Time, an
innovative time series forecasting model that integrates time and frequency
domain features within a Mixture of Experts (MoE) network. Moreover, we use the
pretraining-finetuning paradigm as our training framework to effectively
transfer prior pattern knowledge across pretraining and finetuning datasets
with different periodicity distributions. Our method introduces both frequency
and time cells as experts after attention modules and leverages the MoE routing
mechanism to construct multidimensional sparse representations of input
signals. In experiments on six public benchmarks, MoFE-Time has achieved new
state-of-the-art performance, reducing MSE and MAE by 6.95% and 6.02% compared
to the representative methods Time-MoE. Beyond the existing evaluation
benchmarks, we have developed a proprietary dataset, NEV-sales, derived from
real-world business scenarios. Our method achieves outstanding results on this
dataset, underscoring the effectiveness of the MoFE-Time model in practical
commercial applications.

### 7. [GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models](http://arxiv.org/pdf/2507.06507v1)

Authors: Zhen Yang, Haitao Lin, Jiawei xue, Ziji Zhang

In the past year, Generative Recommendations (GRs) have undergone substantial
advancements, especially in leveraging the powerful sequence modeling and
reasoning capabilities of Large Language Models (LLMs) to enhance overall
recommendation performance. LLM-based GRs are forming a new paradigm that is
distinctly different from discriminative recommendations, showing strong
potential to replace traditional recommendation systems heavily dependent on
complex hand-crafted features. In this paper, we provide a comprehensive survey
aimed at facilitating further research of LLM-based GRs. Initially, we outline
the general preliminaries and application cases of LLM-based GRs. Subsequently,
we introduce the main considerations when LLM-based GRs are applied in real
industrial scenarios. Finally, we explore promising directions for LLM-based
GRs. We hope that this survey contributes to the ongoing advancement of the GR
domain.

### 8. [Towards LLM-based Root Cause Analysis of Hardware Design Failures](http://arxiv.org/pdf/2507.06512v1)

Authors: Siyu Qiu, Muzhi Wang, Raheel Afsharmazayejani, Mohammad Moradi Shahmiri, Benjamin Tan, Hammond Pearce

With advances in large language models (LLMs), new opportunities have emerged
to develop tools that support the digital hardware design process. In this
work, we explore how LLMs can assist with explaining the root cause of design
issues and bugs that are revealed during synthesis and simulation, a necessary
milestone on the pathway towards widespread use of LLMs in the hardware design
process and for hardware security analysis. We find promising results: for our
corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model
reached a correct determination 100% of the time under pass@5 scoring, with
other state of the art models and configurations usually achieving more than
80% performance and more than 90% when assisted with retrieval-augmented
generation.

### 9. [Failure Forecasting Boosts Robustness of Sim2Real Rhythmic Insertion Policies](http://arxiv.org/pdf/2507.06519v1)

Authors: Yuhan Liu, Xinyu Zhang, Haonan Chang, Abdeslam Boularias

This paper addresses the challenges of Rhythmic Insertion Tasks (RIT), where
a robot must repeatedly perform high-precision insertions, such as screwing a
nut into a bolt with a wrench. The inherent difficulty of RIT lies in achieving
millimeter-level accuracy and maintaining consistent performance over multiple
repetitions, particularly when factors like nut rotation and friction introduce
additional complexity. We propose a sim-to-real framework that integrates a
reinforcement learning-based insertion policy with a failure forecasting
module. By representing the wrench's pose in the nut's coordinate frame rather
than the robot's frame, our approach significantly enhances sim-to-real
transferability. The insertion policy, trained in simulation, leverages
real-time 6D pose tracking to execute precise alignment, insertion, and
rotation maneuvers. Simultaneously, a neural network predicts potential
execution failures, triggering a simple recovery mechanism that lifts the
wrench and retries the insertion. Extensive experiments in both simulated and
real-world environments demonstrate that our method not only achieves a high
one-time success rate but also robustly maintains performance over long-horizon
repetitive tasks.

### 10. [Gradientsys: A Multi-Agent LLM Scheduler with ReAct Orchestration](http://arxiv.org/pdf/2507.06520v1)

Authors: Xinyuan Song, Zeyu Wang, Siyi Wu, Tianyu Shi, Lynn Ai

We present Gradientsys, a next-generation multi-agent scheduling framework
that coordinates diverse specialized AI agents using a typed Model-Context
Protocol (MCP) and a ReAct-based dynamic planning loop. At its core,
Gradientsys employs an LLM-powered scheduler for intelligent one-to-many task
dispatch, enabling parallel execution of heterogeneous agents such as PDF
parsers, web search modules, GUI controllers, and web builders. The framework
supports hybrid synchronous/asynchronous execution, respects agent capacity
constraints, and incorporates a robust retry-and-replan mechanism to handle
failures gracefully. To promote transparency and trust, Gradientsys includes an
observability layer streaming real-time agent activity and intermediate
reasoning via Server-Sent Events (SSE). We offer an architectural overview and
evaluate Gradientsys against existing frameworks in terms of extensibility,
scheduling topology, tool reusability, parallelism, and observability.
Experiments on the GAIA general-assistant benchmark show that Gradientsys
achieves higher task success rates with reduced latency and lower API costs
compared to a MinionS-style baseline, demonstrating the strength of its
LLM-driven multi-agent orchestration.

### Hardware Architecture

### 1. [Opto-ViT: Architecting a Near-Sensor Region of Interest-Aware Vision Transformer Accelerator with Silicon Photonics](http://arxiv.org/pdf/2507.07044v1)

Authors: Mehrdad Morsali, Chengwei Zhou, Deniz Najafi, Sreetama Sarkar, Pietro Mercati, Navid Khoshavi, Peter Beerel, Mahdi Nikdast, Gourav Datta, Shaahin Angizi

Vision Transformers (ViTs) have emerged as a powerful architecture for
computer vision tasks due to their ability to model long-range dependencies and
global contextual relationships. However, their substantial compute and memory
demands hinder efficient deployment in scenarios with strict energy and
bandwidth limitations. In this work, we propose OptoViT, the first near-sensor,
region-aware ViT accelerator leveraging silicon photonics (SiPh) for real-time
and energy-efficient vision processing. Opto-ViT features a hybrid
electronic-photonic architecture, where the optical core handles
compute-intensive matrix multiplications using Vertical-Cavity Surface-Emitting
Lasers (VCSELs) and Microring Resonators (MRs), while nonlinear functions and
normalization are executed electronically. To reduce redundant computation and
patch processing, we introduce a lightweight Mask Generation Network (MGNet)
that identifies regions of interest in the current frame and prunes irrelevant
patches before ViT encoding. We further co-optimize the ViT backbone using
quantization-aware training and matrix decomposition tailored for photonic
constraints. Experiments across device fabrication, circuit and architecture
co-design, to classification, detection, and video tasks demonstrate that
OptoViT achieves 100.4 KFPS/W with up to 84% energy savings with less than 1.6%
accuracy loss, while enabling scalable and efficient ViT deployment at the
edge.

### 2. [Towards LLM-based Root Cause Analysis of Hardware Design Failures](http://arxiv.org/pdf/2507.06512v1)

Authors: Siyu Qiu, Muzhi Wang, Raheel Afsharmazayejani, Mohammad Moradi Shahmiri, Benjamin Tan, Hammond Pearce

With advances in large language models (LLMs), new opportunities have emerged
to develop tools that support the digital hardware design process. In this
work, we explore how LLMs can assist with explaining the root cause of design
issues and bugs that are revealed during synthesis and simulation, a necessary
milestone on the pathway towards widespread use of LLMs in the hardware design
process and for hardware security analysis. We find promising results: for our
corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model
reached a correct determination 100% of the time under pass@5 scoring, with
other state of the art models and configurations usually achieving more than
80% performance and more than 90% when assisted with retrieval-augmented
generation.

### 3. [Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs](http://arxiv.org/pdf/2507.06549v1)

Authors: Shan Shen, Dingcheng Yang, Yuyang Xie, Chunyan Pei, Wenjian Yu, Bei Yu

To achieve higher system energy efficiency, SRAM in SoCs is often customized.
The parasitic effects cause notable discrepancies between pre-layout and
post-layout circuit simulations, leading to difficulty in converging design
parameters and excessive design iterations. Is it possible to well predict the
parasitics based on the pre-layout circuit, so as to perform parasitic-aware
pre-layout simulation? In this work, we propose a deep-learning-based 2-stage
model to accurately predict these parasitics in pre-layout stages. The model
combines a Graph Neural Network (GNN) classifier and Multi-Layer Perceptron
(MLP) regressors, effectively managing class imbalance of the net parasitics in
SRAM circuits. We also employ Focal Loss to mitigate the impact of abundant
internal net samples and integrate subcircuit information into the graph to
abstract the hierarchical structure of schematics. Experiments on 4 real SRAM
designs show that our approach not only surpasses the state-of-the-art model in
parasitic prediction by a maximum of 19X reduction of error but also
significantly boosts the simulation process by up to 598X speedup.

### Computational Complexity

### 1. [Efficient Algorithms for Quantum Hashing](http://arxiv.org/pdf/2507.07002v1)

Authors: Ilnar Zinnatullin, Kamil Khadiev

Quantum hashing is a useful technique that allows us to construct
memory-efficient algorithms and secure quantum protocols. First, we present a
circuit that implements the phase form of quantum hashing using $2^{n-1}$ CNOT
gates, where n is the number of control qubits. Our method outperforms existing
approaches and reduces the circuit depth. Second, we propose an algorithm that
provides a trade-off between the number of CNOT gates (and consequently, the
circuit depth) and the precision of rotation angles. This is particularly
important in the context of NISQ (Noisy Intermediate-Scale Quantum) devices,
where hardware-imposed angle precision limit remains a critical constraint.

### Computational Engineering

### 1. [Text to model via SysML: Automated generation of dynamical system computational models from unstructured natural language text via enhanced System Modeling Language diagrams](http://arxiv.org/pdf/2507.06803v1)

Authors: Matthew Anderson Hendricks, Alice Cicirello

This paper contributes to speeding up the design and deployment of
engineering dynamical systems by proposing a strategy for exploiting domain and
expert knowledge for the automated generation of dynamical system computational
model starting from a corpus of document relevant to the dynamical system of
interest and an input document describing the specific system. This strategy is
implemented in five steps and, crucially, it uses system modeling language
diagrams (SysML) to extract accurate information about the dependencies,
attributes, and operations of components. Natural Language Processing (NLP)
strategies and Large Language Models (LLMs) are employed in specific tasks to
improve intermediate outputs of the SySML diagrams automated generation, such
as: list of key nouns; list of extracted relationships; list of key phrases and
key relationships; block attribute values; block relationships; and BDD diagram
generation. The applicability of automated SysML diagram generation is
illustrated with different case studies. The computational models of complex
dynamical systems from SysML diagrams are then obtained via code generation and
computational model generation steps. In the code generation step, NLP
strategies are used for summarization, while LLMs are used for validation only.
The proposed approach is not limited to a specific system, domain, or
computational software. The applicability of the proposed approach is shown via
an end-to-end example from text to model of a simple pendulum, showing improved
performance compared to results yielded by LLMs only.

### 2. [DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models](http://arxiv.org/pdf/2507.06853v1)

Authors: Liang Wang, Yu Rong, Tingyang Xu, Zhenyi Zhong, Zhiyuan Liu, Pengju Wang, Deli Zhao, Qiang Liu, Shu Wu, Liang Wang

Molecular structure elucidation from spectra is a foundational problem in
chemistry, with profound implications for compound identification, synthesis,
and drug development. Traditional methods rely heavily on expert interpretation
and lack scalability. Pioneering machine learning methods have introduced
retrieval-based strategies, but their reliance on finite libraries limits
generalization to novel molecules. Generative models offer a promising
alternative, yet most adopt autoregressive SMILES-based architectures that
overlook 3D geometry and struggle to integrate diverse spectral modalities. In
this work, we present DiffSpectra, a generative framework that directly infers
both 2D and 3D molecular structures from multi-modal spectral data using
diffusion models. DiffSpectra formulates structure elucidation as a conditional
generation process. Its denoising network is parameterized by Diffusion
Molecule Transformer, an SE(3)-equivariant architecture that integrates
topological and geometric information. Conditioning is provided by SpecFormer,
a transformer-based spectral encoder that captures intra- and inter-spectral
dependencies from multi-modal spectra. Extensive experiments demonstrate that
DiffSpectra achieves high accuracy in structure elucidation, recovering exact
structures with 16.01% top-1 accuracy and 96.86% top-20 accuracy through
sampling. The model benefits significantly from 3D geometric modeling,
SpecFormer pre-training, and multi-modal conditioning. These results highlight
the effectiveness of spectrum-conditioned diffusion modeling in addressing the
challenge of molecular structure elucidation. To our knowledge, DiffSpectra is
the first framework to unify multi-modal spectral reasoning and joint 2D/3D
generative modeling for de novo molecular structure elucidation.

### Computational Geometry

### 1. [An Improved Bound for Plane Covering Paths](http://arxiv.org/pdf/2507.06477v1)

Authors: Hugo A. Akitaya, Greg Aloupis, Ahmad Biniaz, Prosenjit Bose, Jean-Lou De Carufel, Cyril Gavoille, John Iacono, Linda Kleist, Michiel Smid, Diane Souvaine, Leonidas Theocharous

A covering path for a finite set $P$ of points in the plane is a polygonal
path such that every point of $P$ lies on a segment of the path. The vertices
of the path need not be at points of $P$. A covering path is plane if its
segments do not cross each other. Let $\pi(n)$ be the minimum number such that
every set of $n$ points in the plane admits a plane covering path with at most
$\pi(n)$ segments. We prove that $\pi(n)\le \lceil6n/7\rceil$. This improves
the previous best-known upper bound of $\lceil 21n/22\rceil$, due to Biniaz
(SoCG 2023). Our proof is constructive and yields a simple $O(n\log n)$-time
algorithm for computing a plane covering path.

### Computation and Language

### 1. [On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](http://arxiv.org/pdf/2507.06489v1)

Authors: Stephen Obadinma, Xiaodan Zhu

Robust verbal confidence generated by large language models (LLMs) is crucial
for the deployment of LLMs to ensure transparency, trust, and safety in
human-AI interactions across many high-stakes applications. In this paper, we
present the first comprehensive study on the robustness of verbal confidence
under adversarial attacks. We introduce a novel framework for attacking verbal
confidence scores through both perturbation and jailbreak-based methods, and
show that these attacks can significantly jeopardize verbal confidence
estimates and lead to frequent answer changes. We examine a variety of
prompting strategies, model sizes, and application domains, revealing that
current confidence elicitation methods are vulnerable and that commonly used
defence techniques are largely ineffective or counterproductive. Our findings
underscore the urgent need to design more robust mechanisms for confidence
expression in LLMs, as even subtle semantic-preserving modifications can lead
to misleading confidence in responses.

### 2. [SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](http://arxiv.org/pdf/2507.06517v1)

Authors: Zicong Tang, Shi Luohe, Zuchao Li, Baoyuan Qi, Guoming Liu, Lefei Zhang, Ping Wang

Large Language Models (LLMs) have achieved impressive accomplishments in
recent years. However, the increasing memory consumption of KV cache has
possessed a significant challenge to the inference system. Eviction methods
have revealed the inherent redundancy within the KV cache, demonstrating its
potential for reduction, particularly in deeper layers. However, KV cache
reduction for shallower layers has been found to be insufficient. Based on our
observation that, the KV cache exhibits a high degree of similarity. Based on
this observation, we proposed a novel KV cache reduction method, SpindleKV,
which balances both shallow and deep layers. For deep layers, we employ an
attention weight based eviction method, while for shallow layers, we apply a
codebook based replacement approach which is learnt by similarity and merging
policy. Moreover, SpindleKV addressed the Grouped-Query Attention (GQA) dilemma
faced by other attention based eviction methods. Experiments on two common
benchmarks with three different LLMs shown that SpindleKV obtained better KV
cache reduction effect compared to baseline methods, while preserving similar
or even better model performance.

### 3. [Large Language Model for Extracting Complex Contract Information in Industrial Scenes](http://arxiv.org/pdf/2507.06539v1)

Authors: Yunyang Cao, Yanjun Li, Silong Dai

This paper proposes a high-quality dataset construction method for complex
contract information extraction tasks in industrial scenarios and fine-tunes a
large language model based on this dataset. Firstly, cluster analysis is
performed on industrial contract texts, and GPT-4 and GPT-3.5 are used to
extract key information from the original contract data, obtaining high-quality
data annotations. Secondly, data augmentation is achieved by constructing new
texts, and GPT-3.5 generates unstructured contract texts from randomly combined
keywords, improving model robustness. Finally, the large language model is
fine-tuned based on the high-quality dataset. Experimental results show that
the model achieves excellent overall performance while ensuring high field
recall and precision and considering parsing efficiency. LoRA, data balancing,
and data augmentation effectively enhance model accuracy and robustness. The
proposed method provides a novel and efficient solution for industrial contract
information extraction tasks.

### 4. [Enhancing Food-Domain Question Answering with a Multimodal Knowledge Graph: Hybrid QA Generation and Diversity Analysis](http://arxiv.org/pdf/2507.06571v1)

Authors: Srihari K B, Pushpak Bhattacharyya

We propose a unified food-domain QA framework that combines a large-scale
multimodal knowledge graph (MMKG) with generative AI. Our MMKG links 13,000
recipes, 3,000 ingredients, 140,000 relations, and 14,000 images. We generate
40,000 QA pairs using 40 templates and LLaVA/DeepSeek augmentation. Joint
fine-tuning of Meta LLaMA 3.1-8B and Stable Diffusion 3.5-Large improves
BERTScore by 16.2\%, reduces FID by 37.8\%, and boosts CLIP alignment by
31.1\%. Diagnostic analyses-CLIP-based mismatch detection (35.2\% to 7.3\%) and
LLaVA-driven hallucination checks-ensure factual and visual fidelity. A hybrid
retrieval-generation strategy achieves 94.1\% accurate image reuse and 85\%
adequacy in synthesis. Our results demonstrate that structured knowledge and
multimodal generation together enhance reliability and diversity in food QA.

### 5. [FuDoBa: Fusing Document and Knowledge Graph-based Representations with Bayesian Optimisation](http://arxiv.org/pdf/2507.06622v1)

Authors: Boshko Koloski, Senja Pollak, Roberto Navigli, Blaž Škrlj

Building on the success of Large Language Models (LLMs), LLM-based
representations have dominated the document representation landscape, achieving
great performance on the document embedding benchmarks. However, the
high-dimensional, computationally expensive embeddings from LLMs tend to be
either too generic or inefficient for domain-specific applications. To address
these limitations, we introduce FuDoBa a Bayesian optimisation-based method
that integrates LLM-based embeddings with domain-specific structured knowledge,
sourced both locally and from external repositories like WikiData. This fusion
produces low-dimensional, task-relevant representations while reducing training
complexity and yielding interpretable early-fusion weights for enhanced
classification performance. We demonstrate the effectiveness of our approach on
six datasets in two domains, showing that when paired with robust AutoML-based
classifiers, our proposed representation learning approach performs on par
with, or surpasses, those produced solely by the proprietary LLM-based
embedding baselines.

### 6. [Checklist Engineering Empowers Multilingual LLM Judges](http://arxiv.org/pdf/2507.06774v1)

Authors: Mohammad Ghiasvand Mohammadkhani, Hamid Beigy

Automated text evaluation has long been a central issue in Natural Language
Processing (NLP). Recently, the field has shifted toward using Large Language
Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While
promising and easily adaptable across tasks, this approach has seen limited
exploration in multilingual contexts. Existing multilingual studies often rely
on proprietary models or require extensive training data for fine-tuning,
raising concerns about cost, time, and efficiency. In this paper, we propose
Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free
framework that uses checklist intuition for multilingual evaluation with an
open-source model. Experiments across multiple languages and three benchmark
datasets, under both pointwise and pairwise settings, show that our method
generally surpasses the baselines and performs on par with the GPT-4o model.

### 7. [Adaptive Termination for Multi-round Parallel Reasoning: An Universal Semantic Entropy-Guided Framework](http://arxiv.org/pdf/2507.06829v1)

Authors: Zenan Xu, Zexuan Qiu, Guanhua Huang, Kun Li, Siheng Li, Chenchen Zhang, Kejiao Li, Qi Yi, Yuhao Jiang, Bo Zhou, Fengzong Lian, Zhanhui Kang

Recent advances in large language models (LLMs) have accelerated progress
toward artificial general intelligence, with inference-time scaling emerging as
a key technique. Contemporary approaches leverage either sequential reasoning
(iteratively extending chains of thought) or parallel reasoning (generating
multiple solutions simultaneously) to scale inference. However, both paradigms
face fundamental limitations: sequential scaling typically relies on arbitrary
token budgets for termination, leading to inefficiency or premature cutoff;
while parallel scaling often lacks coordination among parallel branches and
requires intrusive fine-tuning to perform effectively. In light of these
challenges, we aim to design a flexible test-time collaborative inference
framework that exploits the complementary strengths of both sequential and
parallel reasoning paradigms. Towards this goal, the core challenge lies in
developing an efficient and accurate intrinsic quality metric to assess model
responses during collaborative inference, enabling dynamic control and early
termination of the reasoning trace. To address this challenge, we introduce
semantic entropy (SE), which quantifies the semantic diversity of parallel
model responses and serves as a robust indicator of reasoning quality due to
its strong negative correlation with accuracy...

### 8. [Rethinking Verification for LLM Code Generation: From Generation to Testing](http://arxiv.org/pdf/2507.06920v1)

Authors: Zihan Ma, Taolin Zhang, Maosong Cao, Wenwei Zhang, Minnan Luo, Songyang Zhang, Kai Chen

Large language models (LLMs) have recently achieved notable success in
code-generation benchmarks such as HumanEval and LiveCodeBench. However, a
detailed examination reveals that these evaluation suites often comprise only a
limited number of homogeneous test cases, resulting in subtle faults going
undetected. This not only artificially inflates measured performance but also
compromises accurate reward estimation in reinforcement learning frameworks
utilizing verifiable rewards (RLVR). To address these critical shortcomings, we
systematically investigate the test-case generation (TCG) task by proposing
multi-dimensional metrics designed to rigorously quantify test-suite
thoroughness. Furthermore, we introduce a human-LLM collaborative method
(SAGA), leveraging human programming expertise with LLM reasoning capability,
aimed at significantly enhancing both the coverage and the quality of generated
test cases. In addition, we develop a TCGBench to facilitate the study of the
TCG task. Experiments show that SAGA achieves a detection rate of 90.62% and a
verifier accuracy of 32.58% on TCGBench. The Verifier Accuracy (Verifier Acc)
of the code generation evaluation benchmark synthesized by SAGA is 10.78%
higher than that of LiveCodeBench-v6. These results demonstrate the
effectiveness of our proposed method. We hope this work contributes to building
a scalable foundation for reliable LLM code evaluation, further advancing RLVR
in code generation, and paving the way for automated adversarial test synthesis
and adaptive benchmark integration.

### 9. [Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](http://arxiv.org/pdf/2507.06956v1)

Authors: Sezen Perçin, Xin Su, Qutub Sha Syed, Phillip Howard, Aleksei Kuvshinov, Leo Schwinn, Kay-Ulrich Scholl

Large language models (LLMs) are very costly and inefficient to update with
new information. To address this limitation, retrieval-augmented generation
(RAG) has been proposed as a solution that dynamically incorporates external
knowledge during inference, improving factual consistency and reducing
hallucinations. Despite its promise, RAG systems face practical challenges-most
notably, a strong dependence on the quality of the input query for accurate
retrieval. In this paper, we investigate the sensitivity of different
components in the RAG pipeline to various types of query perturbations. Our
analysis reveals that the performance of commonly used retrievers can degrade
significantly even under minor query variations. We study each module in
isolation as well as their combined effect in an end-to-end question answering
setting, using both general-domain and domain-specific datasets. Additionally,
we propose an evaluation framework to systematically assess the query-level
robustness of RAG pipelines and offer actionable recommendations for
practitioners based on the results of more than 1092 experiments we performed.

### 10. [FRaN-X: FRaming and Narratives-eXplorer](http://arxiv.org/pdf/2507.06974v1)

Authors: Artur Muratov, Hana Fatima Shaikh, Vanshikaa Jani, Tarek Mahmoud, Zhuohan Xie, Daniil Orel, Aaryamonvikram Singh, Yuxia Wang, Aadi Joshi, Hasan Iqbal, Ming Shan Hee, Dhruv Sahnan, Nikolaos Nikolaidis, Purificação Silvano, Dimitar Dimitrov, Roman Yangarber, Ricardo Campos, Alípio Jorge, Nuno Guimarães, Elisa Sartori, Nicolas Stefanovitch, Giovanni Da San Martino, Jakub Piskorski, Preslav Nakov

We present FRaN-X, a Framing and Narratives Explorer that automatically
detects entity mentions and classifies their narrative roles directly from raw
text. FRaN-X comprises a two-stage system that combines sequence labeling with
fine-grained role classification to reveal how entities are portrayed as
protagonists, antagonists, or innocents, using a unique taxonomy of 22
fine-grained roles nested under these three main categories. The system
supports five languages (Bulgarian, English, Hindi, Russian, and Portuguese)
and two domains (the Russia-Ukraine Conflict and Climate Change). It provides
an interactive web interface for media analysts to explore and compare framing
across different sources, tackling the challenge of automatically detecting and
labeling how entities are framed. Our system allows end users to focus on a
single article as well as analyze up to four articles simultaneously. We
provide aggregate level analysis including an intuitive graph visualization
that highlights the narrative a group of articles are pushing. Our system
includes a search feature for users to look up entities of interest, along with
a timeline view that allows analysts to track an entity's role transitions
across different contexts within the article. The FRaN-X system and the trained
models are licensed under an MIT License. FRaN-X is publicly accessible at
https://fran-x.streamlit.app/ and a video demonstration is available at
https://youtu.be/VZVi-1B6yYk.

### Cryptography and Security

### 1. [Vectorised Hashing Based on Bernstein-Rabin-Winograd Polynomials over Prime Order Fields](http://arxiv.org/pdf/2507.06490v1)

Authors: Kaushik Nath, Palash Sarkar

We introduce the new AXU hash function decBRWHash, which is parameterised by
the positive integer $c$ and is based on Bernstein-Rabin-Winograd (BRW)
polynomials. Choosing $c>1$ gives a hash function which can be implemented
using $c$-way single instruction multiple data (SIMD) instructions. We report a
set of very comprehensive hand optimised assembly implementations of
4-decBRWHash using avx2 SIMD instructions available on modern Intel processors.
For comparison, we also report similar carefully optimised avx2 assembly
implementations of polyHash, an AXU hash function based on usual polynomials.
Our implementations are over prime order fields, specifically the primes
$2^{127}-1$ and $2^{130}-5$. For the prime $2^{130}-5$, for avx2
implementations, compared to the famous Poly1305 hash function, 4-decBRWHash is
faster for messages which are a few hundred bytes long and achieves a speed-up
of about 16% for message lengths in a few kilobytes range and improves to a
speed-up of about 23% for message lengths in a few megabytes range.

### 2. [A Survey on Artificial Noise for Physical Layer Security: Opportunities, Technologies, Guidelines, Advances, and Trends](http://arxiv.org/pdf/2507.06500v1)

Authors: Hong Niu, Yue Xiao, Xia Lei, Jiangong Chen, Zhihan Xiao, Mao Li, Chau Yuen

Due to the broadcast nature of wireless communications, physical-layer
security has attracted increasing concerns from both academia and industry.
Artificial noise (AN), as one of the promising physical-layer security
techniques, is capable of utilizing the spatial degree-of-freedom of channels
to effectively enhance the security of wireless communications. In contrast to
other physicallayer security techniques, the key distinguishing feature of AN
is to generate specific interfering signals according to channel
characteristics, increasing the secrecy capacity by reducing the wiretap
channel capacity without affecting the legitimate channel capacity. Hence, this
paper provides the latest survey of AN, including its evolution, modeling,
backgrounds, applications, and future trends. Initially, we introduce the
development, fundamentals, and backgrounds of AN. Subsequently, we highlight a
comprehensive survey of the current state of research on various AN-empowered
scenarios and AN-combined technologies. Finally, we discuss some technical
challenges to tackle for AN-aided wireless security in the future.

### 3. [Subgraph Counting under Edge Local Differential Privacy Based on Noisy Adjacency Matrix](http://arxiv.org/pdf/2507.06508v1)

Authors: Jintao Guo, Ying Zhou, Chao Li, Guixun Luo

When analyzing connection patterns within graphs, subgraph counting serves as
an effective and fundamental approach. Edge-local differential privacy
(edge-LDP) and shuffle model have been employed to achieve subgraph counting
under a privacy-preserving situation. Existing algorithms are plagued by high
time complexity, excessive download costs, low accuracy, or dependence on
trusted third parties. To address the aforementioned challenges, we propose the
Noisy Adjacency Matrix (NAM), which combines differential privacy with the
adjacency matrix of the graph. NAM offers strong versatility and scalability,
making it applicable to a wider range of DP variants, DP mechanisms, and graph
types. Based on NAM, we designed five algorithms (TriOR, TriTR, TriMTR, QuaTR,
and 2STAR) to count three types of subgraphs: triangles, quadrangles, and
2-stars. Theoretical and experimental results demonstrate that in triangle
counting, TriOR maximizes accuracy with reduced time complexity among one-round
algorithms, TriTR achieves optimal accuracy, TriMTR achieves the highest
accuracy under low download costs, and QuaTR stands as the first quadrangle
counting algorithm under pure edge-LDP. We implement edge-LDP for noisy data
via a confidence interval-inspired method, providing DP guarantees on
randomized data. Our 2STAR algorithm achieves the highest accuracy in 2-star
counting and can be derived as a byproduct of two-round triangle or quadrangle
counting algorithms, enabling efficient joint estimation of triangle,
quadrangle, and 2-star counts within two query rounds.

### 4. [Approximating Euler Totient Function using Linear Regression on RSA moduli](http://arxiv.org/pdf/2507.06706v1)

Authors: Gilda Rech Bansimba, Regis F. Babindamana, Beni Blaug N. Ibara

The security of the RSA cryptosystem is based on the intractability of
computing Euler's totient function phi(n) for large integers n. Although
deriving phi(n) deterministically remains computationally infeasible for
cryptographically relevant bit lengths, and machine learning presents a
promising alternative for constructing efficient approximations. In this work,
we explore a machine learning approach to approximate Euler's totient function
phi using linear regression models. We consider a dataset of RSA moduli of 64,
128, 256, 512 and 1024 bits along with their corresponding totient values. The
regression model is trained to capture the relationship between the modulus and
its totient, and tested on unseen samples to evaluate its prediction accuracy.
Preliminary results suggest that phi can be approximated within a small
relative error margin, which may be sufficient to aid in certain classes of RSA
attacks. This research opens a direction for integrating statistical learning
techniques into cryptanalysis, providing insights into the feasibility of
attacking cryptosystems using approximation based strategies.

### 5. [PotentRegion4MalDetect: Advanced Features from Potential Malicious Regions for Malware Detection](http://arxiv.org/pdf/2507.06723v1)

Authors: Rama Krishna Koppanati, Monika Santra, Sateesh Kumar Peddoju

Malware developers exploit the fact that most detection models focus on the
entire binary to extract the feature rather than on the regions of potential
maliciousness. Therefore, they reverse engineer a benign binary and inject
malicious code into it. This obfuscation technique circumvents the malware
detection models and deceives the ML classifiers due to the prevalence of
benign features compared to malicious features. However, extracting the
features from the potential malicious regions enhances the accuracy and
decreases false positives. Hence, we propose a novel model named
PotentRegion4MalDetect that extracts features from the potential malicious
regions. PotentRegion4MalDetect determines the nodes with potential
maliciousness in the partially preprocessed Control Flow Graph (CFG) using the
malicious strings given by StringSifter. Then, it extracts advanced features of
the identified potential malicious regions alongside the features from the
completely preprocessed CFG. The features extracted from the completely
preprocessed CFG mitigate obfuscation techniques that attempt to disguise
malicious content, such as suspicious strings. The experiments reveal that the
PotentRegion4MalDetect requires fewer entries to save the features for all
binaries than the model focusing on the entire binary, reducing memory
overhead, faster computation, and lower storage requirements. These advanced
features give an 8.13% increase in SHapley Additive exPlanations (SHAP)
Absolute Mean and a 1.44% increase in SHAP Beeswarm value compared to those
extracted from the entire binary. The advanced features outperform the features
extracted from the entire binary by producing more than 99% accuracy,
precision, recall, AUC, F1-score, and 0.064% FPR.

### 6. [PenTest2.0: Towards Autonomous Privilege Escalation Using GenAI](http://arxiv.org/pdf/2507.06742v1)

Authors: Haitham S. Al-Sinani, Chris J. Mitchell

Ethical hacking today relies on highly skilled practitioners executing
complex sequences of commands, which is inherently time-consuming, difficult to
scale, and prone to human error. To help mitigate these limitations, we
previously introduced 'PenTest++', an AI-augmented system combining automation
with generative AI supporting ethical hacking workflows. However, a key
limitation of PenTest++ was its lack of support for privilege escalation, a
crucial element of ethical hacking. In this paper we present 'PenTest2.0', a
substantial evolution of PenTest++ supporting automated privilege escalation
driven entirely by Large Language Model reasoning. It also incorporates several
significant enhancements: 'Retrieval-Augmented Generation', including both
one-line and offline modes; 'Chain-of-Thought' prompting for intermediate
reasoning; persistent 'PenTest Task Trees' to track goal progression across
turns; and the optional integration of human-authored hints. We describe how it
operates, present a proof-of-concept prototype, and discuss its benefits and
limitations. We also describe application of the system to a controlled Linux
target, showing it can carry out multi-turn, adaptive privilege escalation. We
explain the rationale behind its core design choices, and provide comprehensive
testing results and cost analysis. Our findings indicate that 'PenTest2.0'
represents a meaningful step toward practical, scalable, AI-automated
penetration testing, whilst highlighting the shortcomings of generative AI
systems, particularly their sensitivity to prompt structure, execution context,
and semantic drift, reinforcing the need for further research and refinement in
this emerging space.
  Keywords: AI, Ethical Hacking, Privilege Escalation, GenAI, ChatGPT, LLM
(Large Language Model), HITL (Human-in-the-Loop)

### 7. [BarkBeetle: Stealing Decision Tree Models with Fault Injection](http://arxiv.org/pdf/2507.06986v1)

Authors: Qifan Wang, Jonas Sander, Minmin Jiang, Thomas Eisenbarth, David Oswald

Machine learning models, particularly decision trees (DTs), are widely
adopted across various domains due to their interpretability and efficiency.
However, as ML models become increasingly integrated into privacy-sensitive
applications, concerns about their confidentiality have grown, particularly in
light of emerging threats such as model extraction and fault injection attacks.
Assessing the vulnerability of DTs under such attacks is therefore important.
In this work, we present BarkBeetle, a novel attack that leverages fault
injection to extract internal structural information of DT models. BarkBeetle
employs a bottom-up recovery strategy that uses targeted fault injection at
specific nodes to efficiently infer feature splits and threshold values. Our
proof-of-concept implementation demonstrates that BarkBeetle requires
significantly fewer queries and recovers more structural information compared
to prior approaches, when evaluated on DTs trained with public UCI datasets. To
validate its practical feasibility, we implement BarkBeetle on a Raspberry Pi
RP2350 board and perform fault injections using the Faultier voltage glitching
tool. As BarkBeetle targets general DT models, we also provide an in-depth
discussion on its applicability to a broader range of tree-based applications,
including data stream classification, DT variants, and cryptography schemes.

### 8. [TELSAFE: Security Gap Quantitative Risk Assessment Framework](http://arxiv.org/pdf/2507.06497v1)

Authors: Sarah Ali Siddiqui, Chandra Thapa, Derui Wang, Rayne Holland, Wei Shao, Seyit Camtepe, Hajime Suzuki, Rajiv Shah

Gaps between established security standards and their practical
implementation have the potential to introduce vulnerabilities, possibly
exposing them to security risks. To effectively address and mitigate these
security and compliance challenges, security risk management strategies are
essential. However, it must adhere to well-established strategies and industry
standards to ensure consistency, reliability, and compatibility both within and
across organizations. In this paper, we introduce a new hybrid risk assessment
framework called TELSAFE, which employs probabilistic modeling for quantitative
risk assessment and eliminates the influence of expert opinion bias. The
framework encompasses both qualitative and quantitative assessment phases,
facilitating effective risk management strategies tailored to the unique
requirements of organizations. A specific use case utilizing Common
Vulnerabilities and Exposures (CVE)-related data demonstrates the framework's
applicability and implementation in real-world scenarios, such as in the
telecommunications industry.

### 9. [A Note on the Walsh Spectrum of Power Residue S-Boxes](http://arxiv.org/pdf/2507.06808v1)

Authors: Matthias Johann Steiner

Let $\mathbb{F}_q$ be a prime field with $q \geq 3$, and let $d, m \geq 1$ be
integers such that $\gcd \left( d, q \right) = 1$ and $m \mid (q - 1)$. In this
paper we bound the absolute values of the Walsh spectrum of S-Boxes $S (x) =
x^d \cdot T \left( x^\frac{q - 1}{m} \right)$, where $T$ is a function with $T
(x) \neq 0$ if $x \neq 0$. Such S-Boxes have been proposed for the
Zero-Knowledge-friendly hash functions Grendel and Polocolo. In particular, we
prove the conjectured correlation of the Polocolo S-Box.

### 10. [The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover](http://arxiv.org/pdf/2507.06850v1)

Authors: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

The rapid adoption of Large Language Model (LLM) agents and multi-agent
systems enables unprecedented capabilities in natural language processing and
generation. However, these systems have introduced unprecedented security
vulnerabilities that extend beyond traditional prompt injection attacks. This
paper presents the first comprehensive evaluation of LLM agents as attack
vectors capable of achieving complete computer takeover through the
exploitation of trust boundaries within agentic AI systems where autonomous
entities interact and influence each other. We demonstrate that adversaries can
leverage three distinct attack surfaces - direct prompt injection, RAG backdoor
attacks, and inter-agent trust exploitation - to coerce popular LLMs (including
GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing
malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals
an alarming vulnerability hierarchy: while 41.2% of models succumb to direct
prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical
82.4% can be compromised through inter-agent trust exploitation. Notably, we
discovered that LLMs which successfully resist direct malicious commands will
execute identical payloads when requested by peer agents, revealing a
fundamental flaw in current multi-agent security models. Our findings
demonstrate that only 5.9% of tested models (1/17) proved resistant to all
attack vectors, with the majority exhibiting context-dependent security
behaviors that create exploitable blind spots. Our findings also highlight the
need to increase awareness and research on the security risks of LLMs, showing
a paradigm shift in cybersecurity threats, where AI tools themselves become
sophisticated attack vectors.

### Computer Vision and Pattern Recognition

### 1. [Mask6D: Masked Pose Priors For 6D Object Pose Estimation](http://arxiv.org/pdf/2507.06486v1)

Authors: Yuechen Xie, Haobo Jiang, Jin Xie

Robust 6D object pose estimation in cluttered or occluded conditions using
monocular RGB images remains a challenging task. One reason is that current
pose estimation networks struggle to extract discriminative, pose-aware
features using 2D feature backbones, especially when the available RGB
information is limited due to target occlusion in cluttered scenes. To mitigate
this, we propose a novel pose estimation-specific pre-training strategy named
Mask6D. Our approach incorporates pose-aware 2D-3D correspondence maps and
visible mask maps as additional modal information, which is combined with RGB
images for the reconstruction-based model pre-training. Essentially, this 2D-3D
correspondence maps a transformed 3D object model to 2D pixels, reflecting the
pose information of the target in camera coordinate system. Meanwhile, the
integrated visible mask map can effectively guide our model to disregard
cluttered background information. In addition, an object-focused pre-training
loss function is designed to further facilitate our network to remove the
background interference. Finally, we fine-tune our pre-trained pose prior-aware
network via conventional pose training strategy to realize the reliable pose
prediction. Extensive experiments verify that our method outperforms previous
end-to-end pose estimation methods.

### 2. [Bilateral Collaboration with Large Vision-Language Models for Open Vocabulary Human-Object Interaction Detection](http://arxiv.org/pdf/2507.06510v1)

Authors: Yupeng Hu, Changxing Ding, Chang Sun, Shaoli Huang, Xiangmin Xu

Open vocabulary Human-Object Interaction (HOI) detection is a challenging
task that detects all <human, verb, object> triplets of interest in an image,
even those that are not pre-defined in the training set. Existing approaches
typically rely on output features generated by large Vision-Language Models
(VLMs) to enhance the generalization ability of interaction representations.
However, the visual features produced by VLMs are holistic and coarse-grained,
which contradicts the nature of detection tasks. To address this issue, we
propose a novel Bilateral Collaboration framework for open vocabulary HOI
detection (BC-HOI). This framework includes an Attention Bias Guidance (ABG)
component, which guides the VLM to produce fine-grained instance-level
interaction features according to the attention bias provided by the HOI
detector. It also includes a Large Language Model (LLM)-based Supervision
Guidance (LSG) component, which provides fine-grained token-level supervision
for the HOI detector by the LLM component of the VLM. LSG enhances the ability
of ABG to generate high-quality attention bias. We conduct extensive
experiments on two popular benchmarks: HICO-DET and V-COCO, consistently
achieving superior performance in the open vocabulary and closed settings. The
code will be released in Github.

### 3. [What Demands Attention in Urban Street Scenes? From Scene Understanding towards Road Safety: A Survey of Vision-driven Datasets and Studies](http://arxiv.org/pdf/2507.06513v1)

Authors: Yaoqi Huang, Julie Stephany Berrio, Mao Shan, Stewart Worrall

Advances in vision-based sensors and computer vision algorithms have
significantly improved the analysis and understanding of traffic scenarios. To
facilitate the use of these improvements for road safety, this survey
systematically categorizes the critical elements that demand attention in
traffic scenarios and comprehensively analyzes available vision-driven tasks
and datasets. Compared to existing surveys that focus on isolated domains, our
taxonomy categorizes attention-worthy traffic entities into two main groups
that are anomalies and normal but critical entities, integrating ten categories
and twenty subclasses. It establishes connections between inherently related
fields and provides a unified analytical framework. Our survey highlights the
analysis of 35 vision-driven tasks and comprehensive examinations and
visualizations of 73 available datasets based on the proposed taxonomy. The
cross-domain investigation covers the pros and cons of each benchmark with the
aim of providing information on standards unification and resource
optimization. Our article concludes with a systematic discussion of the
existing weaknesses, underlining the potential effects and promising solutions
from various perspectives. The integrated taxonomy, comprehensive analysis, and
recapitulatory tables serve as valuable contributions to this rapidly evolving
field by providing researchers with a holistic overview, guiding strategic
resource selection, and highlighting critical research gaps.

### 4. [Concept Unlearning by Modeling Key Steps of Diffusion Process](http://arxiv.org/pdf/2507.06526v1)

Authors: Chaoshuo Zhang, Chenhao Lin, Zhengyu Zhao, Le Yang, Qian Wang, Chao Shen

Text-to-image diffusion models (T2I DMs), represented by Stable Diffusion,
which generate highly realistic images based on textual input, have been widely
used. However, their misuse poses serious security risks. While existing
concept unlearning methods aim to mitigate these risks, they struggle to
balance unlearning effectiveness with generative retainability.To overcome this
limitation, we innovatively propose the Key Step Concept Unlearning (KSCU)
method, which ingeniously capitalizes on the unique stepwise sampling
characteristic inherent in diffusion models during the image generation
process. Unlike conventional approaches that treat all denoising steps equally,
KSCU strategically focuses on pivotal steps with the most influence over the
final outcome by dividing key steps for different concept unlearning tasks and
fine-tuning the model only at those steps. This targeted approach reduces the
number of parameter updates needed for effective unlearning, while maximizing
the retention of the model's generative capabilities.Through extensive
benchmark experiments, we demonstrate that KSCU effectively prevents T2I DMs
from generating undesirable images while better retaining the model's
generative capabilities.Our code will be released.

### 5. [Speak2Sign3D: A Multi-modal Pipeline for English Speech to American Sign Language Animation](http://arxiv.org/pdf/2507.06530v1)

Authors: Kazi Mahathir Rahman, Naveed Imtiaz Nafis, Md. Farhan Sadik, Mohammad Al Rafi, Mehedi Hasan Shahed

Helping deaf and hard-of-hearing people communicate more easily is the main
goal of Automatic Sign Language Translation. Although most past research has
focused on turning sign language into text, doing the reverse, turning spoken
English into sign language animations, has been largely overlooked. That's
because it involves multiple steps, such as understanding speech, translating
it into sign-friendly grammar, and generating natural human motion. In this
work, we introduce a complete pipeline that converts English speech into
smooth, realistic 3D sign language animations. Our system starts with Whisper
to translate spoken English into text. Then, we use a MarianMT machine
translation model to translate that text into American Sign Language (ASL)
gloss, a simplified version of sign language that captures meaning without
grammar. This model performs well, reaching BLEU scores of 0.7714 and 0.8923.
To make the gloss translation more accurate, we also use word embeddings such
as Word2Vec and FastText to understand word meanings. Finally, we animate the
translated gloss using a 3D keypoint-based motion system trained on
Sign3D-WLASL, a dataset we created by extracting body, hand, and face key
points from real ASL videos in the WLASL dataset. To support the gloss
translation stage, we also built a new dataset called BookGlossCorpus-CG, which
turns everyday English sentences from the BookCorpus dataset into ASL gloss
using grammar rules. Our system stitches everything together by smoothly
interpolating between signs to create natural, continuous animations. Unlike
previous works like How2Sign and Phoenix-2014T that focus on recognition or use
only one type of data, our pipeline brings together audio, text, and motion in
a single framework that goes all the way from spoken English to lifelike 3D
sign language animation.

### 6. [ILNet: Trajectory Prediction with Inverse Learning Attention for Enhancing Intention Capture](http://arxiv.org/pdf/2507.06531v1)

Authors: Mingjin Zeng, Nan Ouyang, Wenkang Wan, Lei Ao, Qing Cai, Kai Sheng

Trajectory prediction for multi-agent interaction scenarios is a crucial
challenge. Most advanced methods model agent interactions by efficiently
factorized attention based on the temporal and agent axes. However, this static
and foward modeling lacks explicit interactive spatio-temporal coordination,
capturing only obvious and immediate behavioral intentions. Alternatively, the
modern trajectory prediction framework refines the successive predictions by a
fixed-anchor selection strategy, which is difficult to adapt in different
future environments. It is acknowledged that human drivers dynamically adjust
initial driving decisions based on further assumptions about the intentions of
surrounding vehicles. Motivated by human driving behaviors, this paper proposes
ILNet, a multi-agent trajectory prediction method with Inverse Learning (IL)
attention and Dynamic Anchor Selection (DAS) module. IL Attention employs an
inverse learning paradigm to model interactions at neighboring moments,
introducing proposed intentions to dynamically encode the spatio-temporal
coordination of interactions, thereby enhancing the model's ability to capture
complex interaction patterns. Then, the learnable DAS module is proposed to
extract multiple trajectory change keypoints as anchors in parallel with almost
no increase in parameters. Experimental results show that the ILNet achieves
state-of-the-art performance on the INTERACTION and Argoverse motion
forecasting datasets. Particularly, in challenged interaction scenarios, ILNet
achieves higher accuracy and more multimodal distributions of trajectories over
fewer parameters. Our codes are available at https://github.com/mjZeng11/ILNet.

### 7. [A model-agnostic active learning approach for animal detection from camera traps](http://arxiv.org/pdf/2507.06537v1)

Authors: Thi Thu Thuy Nguyen, Duc Thanh Nguyen

Smart data selection is becoming increasingly important in data-driven
machine learning. Active learning offers a promising solution by allowing
machine learning models to be effectively trained with optimal data including
the most informative samples from large datasets. Wildlife data captured by
camera traps are excessive in volume, requiring tremendous effort in data
labelling and animal detection models training. Therefore, applying active
learning to optimise the amount of labelled data would be a great aid in
enabling automated wildlife monitoring and conservation. However, existing
active learning techniques require that a machine learning model (i.e., an
object detector) be fully accessible, limiting the applicability of the
techniques. In this paper, we propose a model-agnostic active learning approach
for detection of animals captured by camera traps. Our approach integrates
uncertainty and diversity quantities of samples at both the object-based and
image-based levels into the active learning sample selection process. We
validate our approach in a benchmark animal dataset. Experimental results
demonstrate that, using only 30% of the training data selected by our approach,
a state-of-the-art animal detector can achieve a performance of equal or
greater than that with the use of the complete training dataset.

### 8. [Token Bottleneck: One Token to Remember Dynamics](http://arxiv.org/pdf/2507.06543v1)

Authors: Taekyung Kim, Dongyoon Han, Byeongho Heo, Jeongeun Park, Sangdoo Yun

Deriving compact and temporally aware visual representations from dynamic
scenes is essential for successful execution of sequential scene understanding
tasks such as visual tracking and robotic manipulation. In this paper, we
introduce Token Bottleneck (ToBo), a simple yet intuitive self-supervised
learning pipeline that squeezes a scene into a bottleneck token and predicts
the subsequent scene using minimal patches as hints. The ToBo pipeline
facilitates the learning of sequential scene representations by conservatively
encoding the reference scene into a compact bottleneck token during the squeeze
step. In the expansion step, we guide the model to capture temporal dynamics by
predicting the target scene using the bottleneck token along with few target
patches as hints. This design encourages the vision backbone to embed temporal
dependencies, thereby enabling understanding of dynamic transitions across
scenes. Extensive experiments in diverse sequential tasks, including video
label propagation and robot manipulation in simulated environments demonstrate
the superiority of ToBo over baselines. Moreover, deploying our pre-trained
model on physical robots confirms its robustness and effectiveness in
real-world environments. We further validate the scalability of ToBo across
different model scales.

### 9. [Edge-Boundary-Texture Loss: A Tri-Class Generalization of Weighted Binary Cross-Entropy for Enhanced Edge Detection](http://arxiv.org/pdf/2507.06569v1)

Authors: Hao Shu

Edge detection (ED) remains a fundamental task in computer vision, yet its
performance is often hindered by the ambiguous nature of non-edge pixels near
object boundaries. The widely adopted Weighted Binary Cross-Entropy (WBCE) loss
treats all non-edge pixels uniformly, overlooking the structural nuances around
edges and often resulting in blurred predictions. In this paper, we propose the
Edge-Boundary-Texture (EBT) loss, a novel objective that explicitly divides
pixels into three categories, edge, boundary, and texture, and assigns each a
distinct supervisory weight. This tri-class formulation enables more structured
learning by guiding the model to focus on both edge precision and contextual
boundary localization. We theoretically show that the EBT loss generalizes the
WBCE loss, with the latter becoming a limit case. Extensive experiments across
multiple benchmarks demonstrate the superiority of the EBT loss both
quantitatively and perceptually. Furthermore, the consistent use of unified
hyperparameters across all models and datasets, along with robustness to their
moderate variations, indicates that the EBT loss requires minimal fine-tuning
and is easily deployable in practice.

### 10. [MOST: Motion Diffusion Model for Rare Text via Temporal Clip Banzhaf Interaction](http://arxiv.org/pdf/2507.06590v1)

Authors: Yin Wang, Mu li, Zhiying Leng, Frederick W. B. Li, Xiaohui Liang

We introduce MOST, a novel motion diffusion model via temporal clip Banzhaf
interaction, aimed at addressing the persistent challenge of generating human
motion from rare language prompts. While previous approaches struggle with
coarse-grained matching and overlook important semantic cues due to motion
redundancy, our key insight lies in leveraging fine-grained clip relationships
to mitigate these issues. MOST's retrieval stage presents the first formulation
of its kind - temporal clip Banzhaf interaction - which precisely quantifies
textual-motion coherence at the clip level. This facilitates direct,
fine-grained text-to-motion clip matching and eliminates prevalent redundancy.
In the generation stage, a motion prompt module effectively utilizes retrieved
motion clips to produce semantically consistent movements. Extensive
evaluations confirm that MOST achieves state-of-the-art text-to-motion
retrieval and generation performance by comprehensively addressing previous
challenges, as demonstrated through quantitative and qualitative results
highlighting its effectiveness, especially for rare prompts.

### Computers and Society

### 1. [Google Search Advertising after Dobbs v. Jackson](http://arxiv.org/pdf/2507.06640v1)

Authors: Yelena Mejova, Ronald E. Robertson, Catherine A. Gimbrone, Sarah McKetta

Search engines have become the gateway to information, products, and
services, including those concerning healthcare. Access to reproductive health
has been especially complicated in the wake of the 2022 Dobbs v. Jackson
decision by the Supreme Court of the United States, splintering abortion
regulations among the states. In this study, we performed an audit of the
advertisements shown to Google Search users seeking information about abortion
across the United States during the year following the Dobbs decision. We found
that Crisis Pregnancy Centers (CPCs) -- organizations that target women with
unexpected or "crisis" pregnancies, but do not provide abortions -- accounted
for 47% of advertisements, whereas abortion clinics -- for 30%. Advertisements
from CPCs were often returned for queries concerning information and safety.
The type of advertisements returned, however, varied widely within each state,
with Arizona returning the most advertisements from abortion clinics and other
pro-choice organizations, and Minnesota the least. The proportion of pro-choice
vs. anti-choice advertisements returned also varied over time, but estimates
from Staggered Augmented Synthetic Control Methods did not indicate that
changes in advertisement results were attributable to changes in state abortion
laws. Our findings raise questions about the access to accurate medical
information across the U.S. and point to a need for further examination of
search engine advertisement policies and geographical bias.

### 2. [Connecting the Unconnected -- Sentiment Analysis of Field Survey of Internet Connectivity in Emerging Economies](http://arxiv.org/pdf/2507.06827v1)

Authors: Dibakar Das, Barath S Narayan, Aarna Bhammar, Jyotsna Bapat

Internet has significantly improved the quality of citizens across the world.
Though the internet coverage is quite high, 40% of global population do not
have access to broadband internet. This paper presents an analysis of a field
survey of population in some areas of Kathmandu, Nepal, an emerging economy.
This survey was triggered by intermittent severe congestion of internet in
certain areas of the city. People from three different areas were asked about
their present experience of internet usage, its impact on their lives and their
aspirations for the future. Survey pointed to high speed, low cost, reliable
and secure internet as a major aspiration of the respondents. Based on their
inputs, this paper presents a sentiment analysis as well as demographic
information. Keys insights from this analysis shows that overall sentiment to
most queries are positive. The variances of positive sentiments are high
whereas those for negative ones are low. Also, some correlations and clusters
are observed among the attributes though no dominant component exists in the
data.

### 3. [Winning and losing with Artificial Intelligence: What public discourse about ChatGPT tells us about how societies make sense of technological change](http://arxiv.org/pdf/2507.06876v1)

Authors: Adrian Rauchfleisch, Joshua Philip Suarez, Nikka Marie Sales, Andreas Jungherr

Public product launches in Artificial Intelligence can serve as focusing
events for collective attention, surfacing how societies react to technological
change. Social media provide a window into the sensemaking around these events,
surfacing hopes and fears and showing who chooses to engage in the discourse
and when. We demonstrate that public sensemaking about AI is shaped by economic
interests and cultural values of those involved. We analyze 3.8 million tweets
posted by 1.6 million users across 117 countries in response to the public
launch of ChatGPT in 2022. Our analysis shows how economic self-interest,
proxied by occupational skill types in writing, programming, and mathematics,
and national cultural orientations, as measured by Hofstede's individualism,
uncertainty avoidance, and power distance dimensions, shape who speaks, when
they speak, and their stance towards ChatGPT. Roles requiring more technical
skills, such as programming and mathematics, tend to engage earlier and express
more positive stances, whereas writing-centric occupations join later with
greater skepticism. At the cultural level, individualism predicts both earlier
engagement and a more negative stance, and uncertainty avoidance reduces the
prevalence of positive stances but does not delay when users first engage with
ChatGPT. Aggregate sentiment trends mask the dynamics observed in our study.
The shift toward a more critical stance towards ChatGPT over time stems
primarily from the entry of more skeptical voices rather than a change of heart
among early adopters. Our findings underscore the importance of both the
occupational background and cultural context in understanding public reactions
to AI.

### 4. [Do AI tutors empower or enslave learners? Toward a critical use of AI in education](http://arxiv.org/pdf/2507.06878v1)

Authors: Lucile Favero, Juan-Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver

The increasing integration of AI tools in education presents both
opportunities and challenges, particularly regarding the development of the
students' critical thinking skills. This position paper argues that while AI
can support learning, its unchecked use may lead to cognitive atrophy, loss of
agency, emotional risks, and ethical concerns, ultimately undermining the core
goals of education. Drawing on cognitive science and pedagogy, the paper
explores how over-reliance on AI can disrupt meaningful learning, foster
dependency and conformity, undermine the students' self-efficacy, academic
integrity, and well-being, and raise concerns about questionable privacy
practices. It also highlights the importance of considering the students'
perspectives and proposes actionable strategies to ensure that AI serves as a
meaningful support rather than a cognitive shortcut. The paper advocates for an
intentional, transparent, and critically informed use of AI that empowers
rather than diminishes the learner.

### 5. [Exploring LLMs for Predicting Tutor Strategy and Student Outcomes in Dialogues](http://arxiv.org/pdf/2507.06910v1)

Authors: Fareya Ikram, Alexander Scarlatos, Andrew Lan

Tutoring dialogues have gained significant attention in recent years, given
the prominence of online learning and the emerging tutoring abilities of
artificial intelligence (AI) agents powered by large language models (LLMs).
Recent studies have shown that the strategies used by tutors can have
significant effects on student outcomes, necessitating methods to predict how
tutors will behave and how their actions impact students. However, few works
have studied predicting tutor strategy in dialogues. Therefore, in this work we
investigate the ability of modern LLMs, particularly Llama 3 and GPT-4o, to
predict both future tutor moves and student outcomes in dialogues, using two
math tutoring dialogue datasets. We find that even state-of-the-art LLMs
struggle to predict future tutor strategy while tutor strategy is highly
indicative of student outcomes, outlining a need for more powerful methods to
approach this task.

### 6. [Civil Society in the Loop: Feedback-Driven Adaptation of (L)LM-Assisted Classification in an Open-Source Telegram Monitoring Tool](http://arxiv.org/pdf/2507.06734v1)

Authors: Milena Pustet, Elisabeth Steffen, Helena Mihaljević, Grischa Stanjek, Yannis Illies

The role of civil society organizations (CSOs) in monitoring harmful online
content is increasingly crucial, especially as platform providers reduce their
investment in content moderation. AI tools can assist in detecting and
monitoring harmful content at scale. However, few open-source tools offer
seamless integration of AI models and social media monitoring infrastructures.
Given their thematic expertise and contextual understanding of harmful content,
CSOs should be active partners in co-developing technological tools, providing
feedback, helping to improve models, and ensuring alignment with stakeholder
needs and values, rather than as passive 'consumers'. However, collaborations
between the open source community, academia, and civil society remain rare, and
research on harmful content seldom translates into practical tools usable by
civil society actors. This work in progress explores how CSOs can be
meaningfully involved in an AI-assisted open-source monitoring tool of
anti-democratic movements on Telegram, which we are currently developing in
collaboration with CSO stakeholders.

### 7. [Are NFTs Ready to Keep Australian Artists Engaged?](http://arxiv.org/pdf/2507.06926v1)

Authors: Ruiqiang Li, Brian Yecies, Qin Wang, Shiping Chen, Jun Shen

Non-Fungible Tokens (NFTs) offer a promising mechanism to protect Australian
and Indigenous artists' copyright. They represent and transfer the value of
artwork in digital form. Before adopting NFTs to protect Australian artwork, we
in this paper investigate them empericially. We focus on examining the details
of NFT structure. We start from the underlying structure of NFTs to show how
they represent copyright for both artists and production owners, as well as how
they aim to safeguard or secure the value of digital artworks. We then involve
data collection from various types of sources with different storage methods,
including on-chain, centralized, and decentralized systems. Based on both
metadata and artwork content, we present our analysis and discussion on the
following key issues: copyright, security and artist identification. The final
results of the evaluation, unfortnately, show that the NFT is NOT ready to
protect Australian and Indigenous artists' copyright.

### 8. [Exploring Public Perceptions of Generative AI in Libraries: A Social Media Analysis of X Discussions](http://arxiv.org/pdf/2507.07047v1)

Authors: Yuan Li, Teja Mandaloju, Haihua Chen

This study investigates public perceptions of generative artificial
intelligence (GenAI) in libraries through a large-scale analysis of posts on X
(formerly Twitter). Using a mixed-method approach that combines temporal trend
analysis, sentiment classification, and social network analysis, this paper
explores how public discourse around GenAI and libraries has evolved over time,
the emotional tones that dominate the conversation, and the key users or
organizations driving engagement. The findings reveal that discussions are
predominantly negative in tone, with surges linked to concerns about ethics and
intellectual property. Furthermore, social network analysis identifies both
institutional authority and individual bridge users who facilitate cross-domain
engagement. The results in this paper contribute to the growing body of
literature on GenAI in the library and GLAM (Galleries, Libraries, Archives,
and Museums) sectors and offer a real-time, public-facing perspective on the
emerging opportunities and concerns GenAI presents.

### 9. [Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](http://arxiv.org/pdf/2507.06969v1)

Authors: Bogdan Kulynych, Juan Felipe Gomez, Georgios Kaissis, Jamie Hayes, Borja Balle, Flavio du Pin Calmon, Jean Louis Raisaro

Differentially private (DP) mechanisms are difficult to interpret and
calibrate because existing methods for mapping standard privacy parameters to
concrete privacy risks -- re-identification, attribute inference, and data
reconstruction -- are both overly pessimistic and inconsistent. In this work,
we use the hypothesis-testing interpretation of DP ($f$-DP), and determine that
bounds on attack success can take the same unified form across
re-identification, attribute inference, and data reconstruction risks. Our
unified bounds are (1) consistent across a multitude of attack settings, and
(2) tunable, enabling practitioners to evaluate risk with respect to arbitrary
(including worst-case) levels of baseline risk. Empirically, our results are
tighter than prior methods using $\varepsilon$-DP, R\'enyi DP, and concentrated
DP. As a result, calibrating noise using our bounds can reduce the required
noise by 20% at the same risk level, which yields, e.g., more than 15pp
accuracy increase in a text classification task. Overall, this unifying
perspective provides a principled framework for interpreting and calibrating
the degree of protection in DP against specific levels of re-identification,
attribute inference, or data reconstruction risk.

### Databases

### 1. [Interactive Text-to-SQL via Expected Information Gain for Disambiguation](http://arxiv.org/pdf/2507.06467v1)

Authors: Luyu Qiu, Jianing Li, Chi Su, Lei Chen

Relational databases are foundational to numerous domains, including business
intelligence, scientific research, and enterprise systems. However, accessing
and analyzing structured data often requires proficiency in SQL, which is a
skill that many end users lack. With the development of Natural Language
Processing (NLP) technology, the Text-to-SQL systems attempt to bridge this gap
by translating natural language questions into executable SQL queries via an
automated algorithm. Yet, when operating on complex real-world databases, the
Text-to-SQL systems often suffer from ambiguity due to natural ambiguity in
natural language queries. These ambiguities pose a significant challenge for
existing Text-to-SQL translation systems, which tend to commit early to a
potentially incorrect interpretation. To address this, we propose an
interactive Text-to-SQL framework that models SQL generation as a probabilistic
reasoning process over multiple candidate queries. Rather than producing a
single deterministic output, our system maintains a distribution over possible
SQL outputs and seeks to resolve uncertainty through user interaction. At each
interaction step, the system selects a branching decision and formulates a
clarification question aimed at disambiguating that aspect of the query.
Crucially, we adopt a principled decision criterion based on Expected
Information Gain to identify the clarification that will, in expectation, most
reduce the uncertainty in the SQL distribution.

### 2. [QUEST: Query Optimization in Unstructured Document Analysis](http://arxiv.org/pdf/2507.06515v1)

Authors: Zhaoze Sun, Qiyan Deng, Chengliang Chai, Kaisen Jin, Xinyu Guo, Han Han, Ye Yuan, Guoren Wang, Lei Cao

Most recently, researchers have started building large language models (LLMs)
powered data systems that allow users to analyze unstructured text documents
like working with a database because LLMs are very effective in extracting
attributes from documents. In such systems, LLM-based extraction operations
constitute the performance bottleneck of query execution due to the high
monetary cost and slow LLM inference. Existing systems typically borrow the
query optimization principles popular in relational databases to produce query
execution plans, which unfortunately are ineffective in minimizing LLM cost. To
fill this gap, we propose QUEST, which features a bunch of novel optimization
strategies for unstructured document analysis. First, we introduce an
index-based strategy to minimize the cost of each extraction operation. With
this index, QUEST quickly retrieves the text segments relevant to the target
attributes and only feeds them to LLMs. Furthermore, we design an
evidence-augmented retrieval strategy to reduce the possibility of missing
relevant segments. Moreover, we develop an instance-optimized query execution
strategy: because the attribute extraction cost could vary significantly
document by document, QUEST produces different plans for different documents.
For each document, QUEST produces a plan to minimize the frequency of attribute
extraction. The innovations include LLM cost-aware operator ordering strategies
and an optimized join execution approach that transforms joins into filters.
Extensive experiments on 3 real-world datasets demonstrate the superiority of
QUEST, achieving 30%-6x cost savings while improving the F1 score by 10% -27%
compared with state-of-the-art baselines.

### Distributed, Parallel, and Cluster Computing

### 1. [Towards Efficient and Scalable Distributed Vector Search with RDMA](http://arxiv.org/pdf/2507.06653v1)

Authors: Xiangyu Zhi, Meng Chen, Xiao Yan, Baotong Lu, Hui Li, Qianxi Zhang, Qi Chen, James Cheng

Similarity-based vector search facilitates many important applications such
as search and recommendation but is limited by the memory capacity and
bandwidth of a single machine due to large datasets and intensive data read. In
this paper, we present CoTra, a system that scales up vector search for
distributed execution. We observe a tension between computation and
communication efficiency, which is the main challenge for good scalability,
i.e., handling the local vectors on each machine independently blows up
computation as the pruning power of vector index is not fully utilized, while
running a global index over all machines introduces rich data dependencies and
thus extensive communication. To resolve such tension, we leverage the fact
that vector search is approximate in nature and robust to asynchronous
execution. In particular, we run collaborative vector search over the machines
with algorithm-system co-designs including clustering-based data partitioning
to reduce communication, asynchronous execution to avoid communication stall,
and task push to reduce network traffic. To make collaborative search
efficient, we introduce a suite of system optimizations including task
scheduling, communication batching, and storage format. We evaluate CoTra on
real datasets and compare with four baselines. The results show that when using
16 machines, the query throughput of CoTra scales to 9.8-13.4x over a single
machine and is 2.12-3.58x of the best-performing baseline at 0.95 recall@10.

### 2. [Designing Parallel Algorithms for Community Detection using Arachne](http://arxiv.org/pdf/2507.06471v1)

Authors: Fuhuan Li, Zhihui Du, David A. Bader

The rise of graph data in various fields calls for efficient and scalable
community detection algorithms. In this paper, we present parallel
implementations of two widely used algorithms: Label Propagation and Louvain,
specifically designed to leverage the capabilities of Arachne which is a
Python-accessible, open-source framework for large-scale graph analysis. Our
implementations achieve substantial speedups over existing Python-based tools
like NetworkX and igraph, which lack efficient parallelization, and are
competitive with parallel frameworks such as NetworKit. Experimental results
show that Arachne-based methods outperform these baselines, achieving speedups
of up to 710x over NetworkX, 75x over igraph, and 12x over NetworKit.
Additionally, we analyze the scalability of our implementation under varying
thread counts, demonstrating how different phases contribute to overall
performance gains of the parallel Louvain algorithm. Arachne, including our
community detection implementation, is open-source and available at
https://github.com/Bears-R-Us/arkouda-njit .

### 3. [Nexus: Taming Throughput-Latency Tradeoff in LLM Serving via Efficient GPU Sharing](http://arxiv.org/pdf/2507.06608v1)

Authors: Xiaoxiang Shi, Colin Cai, Junjia Du, Zhanda Zhu, Xingda Wei, Zhihao Jia

Current prefill-decode (PD) disaggregation is typically deployed at the level
of entire serving engines, assigning separate GPUs to handle prefill and decode
phases. While effective at reducing latency, this approach demands more
hardware. To improve GPU utilization, Chunked Prefill mixes prefill and decode
requests within the same batch, but introduces phase interference between
prefill and decode.
  While existing PD disaggregation solutions separate the phases across GPUs,
we ask: can the same decoupling be achieved within a single serving engine? The
key challenge lies in managing the conflicting resource requirements of prefill
and decode when they share the same hardware. In this paper, we first show that
chunked prefill requests cause interference with decode requests due to their
distinct requirements for GPU resources. Second, we find that GPU resources
exhibit diminishing returns. Beyond a saturation point, increasing GPU
allocation yields negligible latency improvements. This insight enables us to
split a single GPU's resources and dynamically allocate them to prefill and
decode on the fly, effectively disaggregating the two phases within the same
GPU.
  Across a range of models and workloads, our system Nexus achieves up to 2.2x
higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM. It also
outperforms SGLang with up to 2x higher throughput, 2x lower TTFT, and 1.7x
lower TBT, and achieves 1.4x higher throughput than vLLM-disaggregation using
only half the number of GPUs.

### 4. [Accelerated Spatio-Temporal Bayesian Modeling for Multivariate Gaussian Processes](http://arxiv.org/pdf/2507.06938v1)

Authors: Lisa Gaedke-Merzhäuser, Vincent Maillou, Fernando Rodriguez Avellaneda, Olaf Schenk, Mathieu Luisier, Paula Moraga, Alexandros Nikolaos Ziogas, Håvard Rue

Multivariate Gaussian processes (GPs) offer a powerful probabilistic
framework to represent complex interdependent phenomena. They pose, however,
significant computational challenges in high-dimensional settings, which
frequently arise in spatial-temporal applications. We present DALIA, a highly
scalable framework for performing Bayesian inference tasks on spatio-temporal
multivariate GPs, based on the methodology of integrated nested Laplace
approximations. Our approach relies on a sparse inverse covariance matrix
formulation of the GP, puts forward a GPU-accelerated block-dense approach, and
introduces a hierarchical, triple-layer, distributed memory parallel scheme. We
showcase weak scaling performance surpassing the state-of-the-art by two orders
of magnitude on a model whose parameter space is 8$\times$ larger and measure
strong scaling speedups of three orders of magnitude when running on 496 GH200
superchips on the Alps supercomputer. Applying DALIA to air pollution data from
northern Italy over 48 days, we showcase refined spatial resolutions over the
aggregated pollutant measurements.

### 5. [A Single Merging Suffices: Recovering Server-based Learning Performance in Decentralized Learning](http://arxiv.org/pdf/2507.06542v1)

Authors: Tongtian Zhu, Tianyu Zhang, Mingze Wang, Zhanpeng Zhou, Can Wang

Decentralized learning provides a scalable alternative to traditional
parameter-server-based training, yet its performance is often hindered by
limited peer-to-peer communication. In this paper, we study how communication
should be scheduled over time, including determining when and how frequently
devices synchronize. Our empirical results show that concentrating
communication budgets in the later stages of decentralized training markedly
improves global generalization. Surprisingly, we uncover that fully connected
communication at the final step, implemented by a single global merging, is
sufficient to match the performance of server-based training. We further show
that low communication in decentralized learning preserves the
\textit{mergeability} of local models throughout training. Our theoretical
contributions, which explains these phenomena, are first to establish that the
globally merged model of decentralized SGD can converge faster than centralized
mini-batch SGD. Technically, we novelly reinterpret part of the discrepancy
among local models, which were previously considered as detrimental noise, as
constructive components that accelerate convergence. This work challenges the
common belief that decentralized learning generalizes poorly under data
heterogeneity and limited communication, while offering new insights into model
merging and neural network loss landscapes.

### 6. [SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference](http://arxiv.org/pdf/2507.06567v1)

Authors: Qian Chen, Xianhao Chen, Kaibin Huang

Mixture-of-Experts (MoE) models improve the scalability of large language
models (LLMs) by activating only a small subset of relevant experts per input.
However, the sheer number of expert networks in an MoE model introduces a
significant storage burden for an edge device. To address this challenge, we
consider a scenario where experts are dispersed within an edge network for
distributed inference. Based on the popular Top-$K$ expert selection strategy,
we formulate a latency minimization problem by optimizing expert caching on
edge servers under storage constraints. When $K=1$, the problem reduces to a
monotone submodular maximization problem with knapsack constraints, for which
we design a greedy-based algorithm with a $(1 - 1/e)$-approximation guarantee.
For the general case where $K\geq1$, expert co-activation within the same MoE
layer introduces non-submodularity, causing greedy methods to be ineffective.
To tackle this issue, we propose a successive greedy decomposition method to
decompose the original problem into a series of subproblems, with each being
solved by a dynamic programming approach. Furthermore, we design an accelerated
algorithm based on the max-convolution technique to obtain the approximate
solution with a provable guarantee in polynomial time. Simulation results on
various MoE models demonstrate that our method significantly reduces inference
latency compared to existing baselines.

### 7. [DICE: Data Influence Cascade in Decentralized Learning](http://arxiv.org/pdf/2507.06931v1)

Authors: Tongtian Zhu, Wenhao Li, Can Wang, Fengxiang He

Decentralized learning offers a promising approach to crowdsource data
consumptions and computational workloads across geographically distributed
compute interconnected through peer-to-peer networks, accommodating the
exponentially increasing demands. However, proper incentives are still in
absence, considerably discouraging participation. Our vision is that a fair
incentive mechanism relies on fair attribution of contributions to
participating nodes, which faces non-trivial challenges arising from the
localized connections making influence ``cascade'' in a decentralized network.
To overcome this, we design the first method to estimate \textbf{D}ata
\textbf{I}nfluence \textbf{C}ascad\textbf{E} (DICE) in a decentralized
environment. Theoretically, the framework derives tractable approximations of
influence cascade over arbitrary neighbor hops, suggesting the influence
cascade is determined by an interplay of data, communication topology, and the
curvature of loss landscape. DICE also lays the foundations for applications
including selecting suitable collaborators and identifying malicious behaviors.
Project page is available at https://raiden-zhu.github.io/blog/2025/DICE/.

### Discrete Mathematics

### 1. [Improved Lower Bounds on Multiflow-Multicut Gaps](http://arxiv.org/pdf/2507.06576v1)

Authors: Sina Kalantarzadeh, Nikhil Kumar

Given a set of source-sink pairs, the maximum multiflow problem asks for the
maximum total amount of flow that can be feasibly routed between them. The
minimum multicut, a dual problem to multiflow, seeks the minimum-cost set of
edges whose removal disconnects all the source-sink pairs. It is easy to see
that the value of the minimum multicut is at least that of the maximum
multiflow, and their ratio is called the multiflow-multicut gap. The classical
max-flow min-cut theorem states that when there is only one source-sink pair,
the gap is exactly one. However, in general, it is well known that this gap can
be arbitrarily large. In this paper, we study this gap for classes of planar
graphs and establish improved lower bound results. In particular, we show that
this gap is at least $\frac{20}{9}$ for the class of planar graphs, improving
upon the decades-old lower bound of 2. More importantly, we develop new
techniques for proving such a lower bound, which may be useful in other
settings as well.

### 2. [On Construction of Approximate Real Mutually Unbiased Bases for an infinite class of dimensions $d \not\equiv 0 \bmod 4$](http://arxiv.org/pdf/2507.07028v1)

Authors: Ajeet Kumar, Rakesh Kumar, Subhamoy Maitra, Uddipto Mandal

It is known that real Mutually Unbiased Bases (MUBs) do not exist for any
dimension $d > 2$ which is not divisible by 4. Thus, the next combinatorial
question is how one can construct Approximate Real MUBs (ARMUBs) in this
direction with encouraging parameters. In this paper, for the first time, we
show that it is possible to construct $> \lceil \sqrt{d} \rceil$ many ARMUBs
for certain odd dimensions $d$ of the form $d = (4n-t)s$, $t = 1, 2, 3$, where
$n$ is a natural number and $s$ is an odd prime power. Our method exploits any
available $4n \times 4n$ real Hadamard matrix $H_{4n}$ (conjectured to be true)
and uses this to construct an orthogonal matrix ${Y}_{4n-t}$ of size $(4n - t)
\times (4n - t)$, such that the absolute value of each entry varies a little
from $\frac{1}{4n-t}$. In our construction, the absolute value of the inner
product between any pair of basis vectors from two different ARMUBs will be
$\leq \frac{1}{\sqrt{d}}(1 + O(d^{-\frac{1}{2}})) < 2$, for proper choices of
parameters, the class of dimensions $d$ being infinitely large.

### 3. [Real-time Optimization of Transport Chains for Single Wagon Load Railway Transport](http://arxiv.org/pdf/2507.06621v1)

Authors: Carsten Moldenhauer, Philipp Germann, Cedric Heimhofer, Caroline Spieckermann, Andreas Andresen

The freight branch of the Swiss national railways, SBB Cargo, offers
customers to ship single or few wagons within its wagon load transportation
system (WLV). In this system, wagons travel along a transport chain which is a
sequence of consecutive trains. Recently, SBB Cargo redesigned its IT systems
and renewed the computation of these transport chains. This paper describes the
main design decisions and technical details: data structures, search
algorithms, mathematical optimization of throughput in the real-time setting,
and some selected details for making the algorithms work in the operational
software. We also comment on the employed technology stack and finally
demonstrate some performance metrics from running operations.

### 4. [The Integrality Gap of the Traveling Salesman Problem is $4/3$ if the LP Solution Has at Most $n+6$ Non-zero Components](http://arxiv.org/pdf/2507.07003v1)

Authors: Tullio Villa, Eleonora Vercesi, Janos Barta, Monaldo Mastrolilli

In this paper, we address the classical Dantzig-Fulkerson-Johnson formulation
of the metric Traveling Salesman Problem and study the integrality gap of its
linear relaxation, namely the Subtour Elimination Problem (SEP). This
integrality gap is conjectured to be $4/3$. We prove that, when solving a
problem on $n$ nodes, if the optimal SEP solution has at most $n+6$ non-zero
components, then the conjecture is true. To establish this result, we consider,
for a given integer $k$, the infinite family $F_k$ which gathers, among all the
vertices of all the SEP polytopes for $n \in \mathbb{N}$, the ones with exactly
$n+k$ non-zero components. Then, we introduce a procedure that reduces the
description of $F_k$ to a finite set, and we present the Gap-Bounding
algorithm, which provides provable upper bounds on the integrality gap for
entire families $F_k$. The application of the Gap-Bounding algorithm for $k
\leq 6$ yields a computer-aided proof that the conjectured bound holds in this
case.

### Data Structures and Algorithms

### 1. [Faster Algorithms for $(2k-1)$-Stretch Distance Oracles](http://arxiv.org/pdf/2507.06721v1)

Authors: Avi Kadria, Liam Roditty

Let $G=(V, E)$ be an undirected $n$-vertices $m$-edges graph with
non-negative edge weights. In this paper, we present three new algorithms for
constructing a $(2k-1)$-stretch distance oracle with $O(n^{1+\frac{1}{k}})$
space. The first algorithm runs in $\Ot(\max(n^{1+2/k},
m^{1-\frac{1}{k-1}}n^{\frac{2}{k-1}}))$ time, and improves upon the
$\Ot(\min(mn^{\frac{1}{k}},n^2))$ time of Thorup and Zwick [STOC 2001, JACM
2005] and Baswana and Kavitha [FOCS 2006, SICOMP 2010], for every $k > 2$ and
$m=\Omega(n^{1+\frac{1}{k}+\eps})$. This yields the first truly subquadratic
time construction for every $2 < k < 6$, and nearly resolves the open problem
posed by Wulff-Nilsen [SODA 2012] on the existence of such constructions.
  The two other algorithms have a running time of the form $\Ot(m+n^{1+f(k)})$,
which is near linear in $m$ if $m=\Omega(n^{1+f(k)})$, and therefore optimal in
such graphs. One algorithm runs in $\Ot(m+n^{\frac32+\frac{3}{4k-6}})$-time,
which improves upon the $\Ot(n^2)$-time algorithm of Baswana and Kavitha [FOCS
2006, SICOMP 2010], for $3 < k < 6$, and upon the
$\Ot(m+n^{\frac{3}{2}+\frac{2}{k}+O(k^{-2})})$-time algorithm of Wulff-Nilsen
[SODA 2012], for every $k\geq 6$. This is the first linear time algorithm for
constructing a $7$-stretch distance oracle and a $9$-stretch distance oracle,
for graphs with truly subquadratic density.\footnote{with $m=n^{2-\eps}$ for
some $\eps > 0$.} The other algorithm runs in
$\Ot(\sqrt{k}m+kn^{1+\frac{2\sqrt{2}}{\sqrt{k}}})$ time, (and hence relevant
only for $k\ge 16$), and improves upon the
$\Ot(\sqrt{k}m+kn^{1+\frac{2\sqrt{6}}{\sqrt{k}}+O(k^{-1})})$ time algorithm of
Wulff-Nilsen [SODA 2012] (which is relevant only for $k\ge 96$). ...

### 2. [Faster Estimation of the Average Degree of a Graph Using Random Edges and Structural Queries](http://arxiv.org/pdf/2507.06925v1)

Authors: Lorenzo Beretta, Deeparnab Chakrabarty, C. Seshadhri

We revisit the problem of designing sublinear algorithms for estimating the
average degree of an $n$-vertex graph. The standard access model for graphs
allows for the following queries: sampling a uniform random vertex, the degree
of a vertex, sampling a uniform random neighbor of a vertex, and ``pair
queries'' which determine if a pair of vertices form an edge. In this model,
original results [Goldreich-Ron, RSA 2008; Eden-Ron-Seshadhri, SIDMA 2019] on
this problem prove that the complexity of getting
$(1+\varepsilon)$-multiplicative approximations to the average degree, ignoring
$\varepsilon$-dependencies, is $\Theta(\sqrt{n})$. When random edges can be
sampled, it is known that the average degree can estimated in
$\widetilde{O}(n^{1/3})$ queries, even without pair queries
[Motwani-Panigrahy-Xu, ICALP 2007; Beretta-Tetek, TALG 2024].
  We give a nearly optimal algorithm in the standard access model with random
edge samples. Our algorithm makes $\widetilde{O}(n^{1/4})$ queries exploiting
the power of pair queries. We also analyze the ``full neighborhood access"
model wherein the entire adjacency list of a vertex can be obtained with a
single query; this model is relevant in many practical applications. In a
weaker version of this model, we give an algorithm that makes
$\widetilde{O}(n^{1/5})$ queries. Both these results underscore the power of
{\em structural queries}, such as pair queries and full neighborhood access
queries, for estimating the average degree. We give nearly matching lower
bounds, ignoring $\varepsilon$-dependencies, for all our results.
  So far, almost all algorithms for estimating average degree assume that the
number of vertices, $n$, is known. Inspired by [Beretta-Tetek, TALG 2024], we
study this problem when $n$ is unknown and show that structural queries do not
help in estimating average degree in this setting.

### 3. [Designing Parallel Algorithms for Community Detection using Arachne](http://arxiv.org/pdf/2507.06471v1)

Authors: Fuhuan Li, Zhihui Du, David A. Bader

The rise of graph data in various fields calls for efficient and scalable
community detection algorithms. In this paper, we present parallel
implementations of two widely used algorithms: Label Propagation and Louvain,
specifically designed to leverage the capabilities of Arachne which is a
Python-accessible, open-source framework for large-scale graph analysis. Our
implementations achieve substantial speedups over existing Python-based tools
like NetworkX and igraph, which lack efficient parallelization, and are
competitive with parallel frameworks such as NetworKit. Experimental results
show that Arachne-based methods outperform these baselines, achieving speedups
of up to 710x over NetworkX, 75x over igraph, and 12x over NetworKit.
Additionally, we analyze the scalability of our implementation under varying
thread counts, demonstrating how different phases contribute to overall
performance gains of the parallel Louvain algorithm. Arachne, including our
community detection implementation, is open-source and available at
https://github.com/Bears-R-Us/arkouda-njit .

### 4. [Prediction-Augmented Mechanism Design for Weighted Facility Location](http://arxiv.org/pdf/2507.06509v1)

Authors: Yangguang Shi, Zhenyu Xue

Facility location is fundamental in operations research, mechanism design,
and algorithmic game theory, with applications ranging from urban
infrastructure planning to distributed systems. Recent research in this area
has focused on augmenting classic strategyproof mechanisms with predictions to
achieve an improved performance guarantee against the uncertainty under the
strategic environment. Previous work has been devoted to address the trade-off
obstacle of balancing the consistency (near-optimality under accurate
predictions) and robustness (bounded inefficiency under poor predictions)
primarily in the unweighted setting, assuming that all agents have the same
importance. However, this assumption may not be true in some practical
scenarios, leading to research of weighted facility location problems.
  The major contribution of the current work is to provide a prediction
augmented algorithmic framework for balancing the consistency and robustness
over strategic agents with non-uniform weights. In particular, through a
reduction technique that identifies a subset of \emph{representative} instances
and maps the other given locations to the representative ones, we prove that
there exists a \emph{strategyproof} mechanism achieving a bounded consistency
guarantee of $\frac{\sqrt{(1+c)^2W^2_{\min}+(1-c)^2W^2_{\max}}}{(1+c)W_{\min}}$
and a bounded robustness guarantee of
$\frac{\sqrt{(1-c)^2W^2_{\min}+(1+c)^2W^2_{\max}}}{(1-c)W_{\min}}$ in weighted
settings, where $c$ can be viewed as a parameter to make a trade-off between
the consistency and robustness and $W_{\min}$ and $W_{\max}$ denote the minimum
and maximum agents' weight. We also proved that there is no strategyproof
deterministic mechanism that reach $1$-consistency and $O\left( n \cdot
\frac{W_{\max}}{W_{\min}} \right)$-robustness in weighted FLP, even with fully
predictions of all agents.

### Emerging Technologies

### 1. [InvestAlign: Overcoming Data Scarcity in Aligning Large Language Models with Investor Decision-Making Processes under Herd Behavior](http://arxiv.org/pdf/2507.06528v1)

Authors: Huisheng Wang, Zhuoshi Pan, Hangjing Zhang, Mingxiao Liu, Hanqing Gao, H. Vicky Zhao

Aligning Large Language Models (LLMs) with investor decision-making processes
under herd behavior is a critical challenge in behavioral finance, which
grapples with a fundamental limitation: the scarcity of real-user data needed
for Supervised Fine-Tuning (SFT). While SFT can bridge the gap between LLM
outputs and human behavioral patterns, its reliance on massive authentic data
imposes substantial collection costs and privacy risks. We propose InvestAlign,
a novel framework that constructs high-quality SFT datasets by leveraging
theoretical solutions to similar and simple optimal investment problems rather
than complex scenarios. Our theoretical analysis demonstrates that training
LLMs with InvestAlign-generated data achieves faster parameter convergence than
using real-user data, suggesting superior learning efficiency. Furthermore, we
develop InvestAgent, an LLM agent fine-tuned with InvestAlign, which
demonstrates significantly closer alignment to real-user data than pre-SFT
models in both simple and complex investment problems. This highlights our
proposed InvestAlign as a promising approach with the potential to address
complex optimal investment problems and align LLMs with investor
decision-making processes under herd behavior. Our code is publicly available
at https://github.com/thu-social-network-research-group/InvestAlign.

### 2. [Illuminating the Future: Nanophotonics for Future Green Technologies, Precision Healthcare, and Optical Computing](http://arxiv.org/pdf/2507.06587v1)

Authors: Osama M. Halawa, Esraa Ahmed, Malk M. Abdelrazek, Yasser M. Nagy, Omar A. M. Abdelraouf

Nanophotonics, an interdisciplinary field merging nanotechnology and
photonics, has enabled transformative advancements across diverse sectors
including green energy, biomedicine, and optical computing. This review
comprehensively examines recent progress in nanophotonic principles and
applications, highlighting key innovations in material design, device
engineering, and system integration. In renewable energy, nanophotonic allows
light-trapping nanostructures and spectral control in perovskite solar cells,
concentrating solar power, and thermophotovoltaics. That have significantly
enhanced solar conversion efficiencies, approaching theoretical limits. For
biosensing, nanophotonic platforms achieve unprecedented sensitivity in
detecting biomolecules, pathogens, and pollutants, enabling real-time
diagnostics and environmental monitoring. Medical applications leverage
tailored light-matter interactions for precision photothermal therapy,
image-guided surgery, and early disease detection. Furthermore, nanophotonics
underpins next-generation optical neural networks and neuromorphic computing,
offering ultra-fast, energy-efficient alternatives to von Neumann
architectures. Despite rapid growth, challenges in scalability, fabrication
costs, and material stability persist. Future advancements will rely on novel
materials, AI-driven design optimization, and multidisciplinary approaches to
enable scalable, low-cost deployment. This review summarizes recent progress
and highlights future trends, including novel material systems,
multidisciplinary approaches, and enhanced computational capabilities, to pave
the way for transformative applications in this rapidly evolving field.

### 3. [Are NFTs Ready to Keep Australian Artists Engaged?](http://arxiv.org/pdf/2507.06926v1)

Authors: Ruiqiang Li, Brian Yecies, Qin Wang, Shiping Chen, Jun Shen

Non-Fungible Tokens (NFTs) offer a promising mechanism to protect Australian
and Indigenous artists' copyright. They represent and transfer the value of
artwork in digital form. Before adopting NFTs to protect Australian artwork, we
in this paper investigate them empericially. We focus on examining the details
of NFT structure. We start from the underlying structure of NFTs to show how
they represent copyright for both artists and production owners, as well as how
they aim to safeguard or secure the value of digital artworks. We then involve
data collection from various types of sources with different storage methods,
including on-chain, centralized, and decentralized systems. Based on both
metadata and artwork content, we present our analysis and discussion on the
following key issues: copyright, security and artist identification. The final
results of the evaluation, unfortnately, show that the NFT is NOT ready to
protect Australian and Indigenous artists' copyright.

### 4. [No physics required! A visual-based introduction to GKP qubits for computer scientists](http://arxiv.org/pdf/2507.06943v1)

Authors: Richard A. Wolf, Pavithran Iyer

With the significance of continuous-variable quantum computing increasing
thanks to the achievements of light-based quantum hardware, making it available
to learner audiences outside physics has been an important yet seldom-tackled
challenge. Similarly, the rising focus on fault-tolerant quantum computing has
shed light on quantum error correction schemes, turning it into the locus of
attention for industry and academia alike. In this paper, we explore the widely
adopted framework of quantum error correction based on continuous variable
systems and suggest a guide on building a self-contained learning session
targeting the famous Gottesman-Kitaev-Preskill (GKP) code through its geometric
intuition.

### 5. [Optimizing Cognitive Networks: Reinforcement Learning Meets Energy Harvesting Over Cascaded Channels](http://arxiv.org/pdf/2507.06981v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

This paper presents a reinforcement learning (RL) based approach to improve
the physical layer security (PLS) of an underlay cognitive radio network (CRN)
over cascaded channels. These channels are utilized in highly mobile networks
such as cognitive vehicular networks (CVN). In addition, an eavesdropper aims
to intercept the communications between secondary users (SUs). The SU receiver
has full-duplex and energy harvesting capabilities to generate jamming signals
to confound the eavesdropper and enhance security. Moreover, the SU transmitter
extracts energy from ambient radio frequency signals in order to power
subsequent transmissions to its intended receiver. To optimize the privacy and
reliability of the SUs in a CVN, a deep Q-network (DQN) strategy is utilized
where multiple DQN agents are required such that an agent is assigned at each
SU transmitter. The objective for the SUs is to determine the optimal
transmission power and decide whether to collect energy or transmit messages
during each time period in order to maximize their secrecy rate. Thereafter, we
propose a DQN approach to maximize the throughput of the SUs while respecting
the interference threshold acceptable at the receiver of the primary user.
According to our findings, our strategy outperforms two other baseline
strategies in terms of security and reliability.

### 6. [Maximizing Reliability in Overlay Radio Networks with Time Switching and Power Splitting Energy Harvesting](http://arxiv.org/pdf/2507.06983v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

Cognitive radio networks (CRNs) are acknowledged for their ability to tackle
the issue of spectrum under-utilization. In the realm of CRNs, this paper
investigates the energy efficiency issue and addresses the critical challenge
of optimizing system reliability for overlay CRN access mode. Randomly
dispersed secondary users (SUs) serving as relays for primary users (PUs) are
considered, in which one of these relays is designated to harvest energy
through the time switching-energy harvesting (EH) protocol. Moreover, this
relay amplifies-and-forwards (AF) the PU's messages and broadcasts them along
with its own across cascaded $\kappa$-$\mu$ fading channels. The power
splitting protocol is another EH approach utilized by the SU and PU receivers
to enhance the amount of energy in their storage devices. In addition, the SU
transmitters and the SU receiver are deployed with multiple antennas for
reception and apply the maximal ratio combining approach. The outage
probability is utilized to assess both networks' reliability. Then, an energy
efficiency evaluation is performed to determine the effectiveness of EH on the
system. Finally, an optimization problem is provided with the goal of
maximizing the data rate of the SUs by optimizing the time switching and the
power allocation parameters of the SU relay.

### 7. [Federated Learning-based MARL for Strengthening Physical-Layer Security in B5G Networks](http://arxiv.org/pdf/2507.06997v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

This paper explores the application of a federated learning-based multi-agent
reinforcement learning (MARL) strategy to enhance physical-layer security (PLS)
in a multi-cellular network within the context of beyond 5G networks. At each
cell, a base station (BS) operates as a deep reinforcement learning (DRL) agent
that interacts with the surrounding environment to maximize the secrecy rate of
legitimate users in the presence of an eavesdropper. This eavesdropper attempts
to intercept the confidential information shared between the BS and its
authorized users. The DRL agents are deemed to be federated since they only
share their network parameters with a central server and not the private data
of their legitimate users. Two DRL approaches, deep Q-network (DQN) and
Reinforce deep policy gradient (RDPG), are explored and compared. The results
demonstrate that RDPG converges more rapidly than DQN. In addition, we
demonstrate that the proposed method outperforms the distributed DRL approach.
Furthermore, the outcomes illustrate the trade-off between security and
complexity.

### Formal Languages and Automata Theory

### 1. [Stochastic Alignments: Matching an Observed Trace to Stochastic Process Models](http://arxiv.org/pdf/2507.06472v1)

Authors: Tian Li, Artem Polyvyanyy, Sander J. J. Leemans

Process mining leverages event data extracted from IT systems to generate
insights into the business processes of organizations. Such insights benefit
from explicitly considering the frequency of behavior in business processes,
which is captured by stochastic process models. Given an observed trace and a
stochastic process model, conventional alignment-based conformance checking
techniques face a fundamental limitation: They prioritize matching the trace to
a model path with minimal deviations, which may, however, lead to selecting an
unlikely path. In this paper, we study the problem of matching an observed
trace to a stochastic process model by identifying a likely model path with a
low edit distance to the trace. We phrase this as an optimization problem and
develop a heuristic-guided path-finding algorithm to solve it. Our open-source
implementation demonstrates the feasibility of the approach and shows that it
can provide new, useful diagnostic insights for analysts.

### Graphics

### 1. [Assessing Learned Models for Phase-only Hologram Compression](http://arxiv.org/pdf/2507.06646v1)

Authors: Zicong Peng, Yicheng Zhan, Josef Spjut, Kaan Akşit

We evaluate the performance of four common learned models utilizing INR and
VAE structures for compressing phase-only holograms in holographic displays.
The evaluated models include a vanilla MLP, SIREN, and FilmSIREN, with TAESD as
the representative VAE model. Our experiments reveal that a pretrained image
VAE, TAESD, with 2.2M parameters struggles with phase-only hologram
compression, revealing the need for task-specific adaptations. Among the INRs,
SIREN with 4.9k parameters achieves %40 compression with high quality in the
reconstructed 3D images (PSNR = 34.54 dB). These results emphasize the
effectiveness of INRs and identify the limitations of pretrained image
compression VAEs for hologram compression task.

### 2. [3D-Generalist: Self-Improving Vision-Language-Action Models for Crafting 3D Worlds](http://arxiv.org/pdf/2507.06484v1)

Authors: Fan-Yun Sun, Shengguang Wu, Christian Jacobsen, Thomas Yim, Haoming Zou, Alex Zook, Shangru Li, Yu-Hsin Chou, Ethem Can, Xunlei Wu, Clemens Eppner, Valts Blukis, Jonathan Tremblay, Jiajun Wu, Stan Birchfield, Nick Haber

Despite large-scale pretraining endowing models with language and vision
reasoning capabilities, improving their spatial reasoning capability remains
challenging due to the lack of data grounded in the 3D world. While it is
possible for humans to manually create immersive and interactive worlds through
3D graphics, as seen in applications such as VR, gaming, and robotics, this
process remains highly labor-intensive. In this paper, we propose a scalable
method for generating high-quality 3D environments that can serve as training
data for foundation models. We recast 3D environment building as a sequential
decision-making problem, employing Vision-Language-Models (VLMs) as policies
that output actions to jointly craft a 3D environment's layout, materials,
lighting, and assets. Our proposed framework, 3D-Generalist, trains VLMs to
generate more prompt-aligned 3D environments via self-improvement fine-tuning.
We demonstrate the effectiveness of 3D-Generalist and the proposed training
strategy in generating simulation-ready 3D environments. Furthermore, we
demonstrate its quality and scalability in synthetic data generation by
pretraining a vision foundation model on the generated data. After fine-tuning
the pre-trained model on downstream tasks, we show that it surpasses models
pre-trained on meticulously human-crafted synthetic data and approaches results
achieved with real data orders of magnitude larger.

### 3. [Better frame rates or better visuals? An early report of Esports player practice in Dota 2](http://arxiv.org/pdf/2507.06790v1)

Authors: Arjun Madhusudan, Benjamin Watson

Esports athletes often reduce visual quality to improve latency and frame
rate, and increase their in-game performance. Little research has examined the
effects of this visuo-spatial tradeoff on performance, but we could find no
work studying how players manage this tradeoff in practice. This paper is an
initial examination of this question in the game Dota 2. First, we gather the
game configuration data of Dota 2 players in a small survey. We learn that
players do limit visual detail, particularly by turning off VSYNC, which
removes rendering/display synchronization delay but permits visual "tearing".
Second, we survey the intent of those same players with a few subjective
questions. Player intent matches configuration practice. While our sampling of
Dota 2 players may not be representative, our survey does reveal suggestive
trends that lay the groundwork for future, more rigorous and larger surveys.
Such surveys can help new players adapt to the game more quickly, encourage
researchers to investigate the relative importance of temporal and visual
detail, and justify design effort by developers in "low visual" game
configurations.

### 4. [Enhancing non-Rigid 3D Model Deformations Using Mesh-based Gaussian Splatting](http://arxiv.org/pdf/2507.07000v1)

Authors: Wijayathunga W. M. R. D. B

We propose a novel framework that enhances non-rigid 3D model deformations by
bridging mesh representations with 3D Gaussian splatting. While traditional
Gaussian splatting delivers fast, real-time radiance-field rendering, its
post-editing capabilities and support for large-scale, non-rigid deformations
remain limited. Our method addresses these challenges by embedding Gaussian
kernels directly onto explicit mesh surfaces. This allows the mesh's inherent
topological and geometric priors to guide intuitive editing operations -- such
as moving, scaling, and rotating individual 3D components -- and enables
complex deformations like bending and stretching. This work paves the way for
more flexible 3D content-creation workflows in applications spanning virtual
reality, character animation, and interactive design.

### 5. [FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation](http://arxiv.org/pdf/2507.06523v1)

Authors: Liqiang Jing, Viet Lai, Seunghyun Yoon, Trung Bui, Xinya Du

Video Multimodal Large Language Models (VideoMLLMs) have achieved remarkable
progress in both Video-to-Text and Text-to-Video tasks. However, they often
suffer fro hallucinations, generating content that contradicts the visual
input. Existing evaluation methods are limited to one task (e.g., V2T) and also
fail to assess hallucinations in open-ended, free-form responses. To address
this gap, we propose FIFA, a unified FaIthFulness evAluation framework that
extracts comprehensive descriptive facts, models their semantic dependencies
via a Spatio-Temporal Semantic Dependency Graph, and verifies them using
VideoQA models. We further introduce Post-Correction, a tool-based correction
framework that revises hallucinated content. Extensive experiments demonstrate
that FIFA aligns more closely with human judgment than existing evaluation
methods, and that Post-Correction effectively improves factual consistency in
both text and video generation.

### Computer Science and Game Theory

### 1. [Prediction-Augmented Mechanism Design for Weighted Facility Location](http://arxiv.org/pdf/2507.06509v1)

Authors: Yangguang Shi, Zhenyu Xue

Facility location is fundamental in operations research, mechanism design,
and algorithmic game theory, with applications ranging from urban
infrastructure planning to distributed systems. Recent research in this area
has focused on augmenting classic strategyproof mechanisms with predictions to
achieve an improved performance guarantee against the uncertainty under the
strategic environment. Previous work has been devoted to address the trade-off
obstacle of balancing the consistency (near-optimality under accurate
predictions) and robustness (bounded inefficiency under poor predictions)
primarily in the unweighted setting, assuming that all agents have the same
importance. However, this assumption may not be true in some practical
scenarios, leading to research of weighted facility location problems.
  The major contribution of the current work is to provide a prediction
augmented algorithmic framework for balancing the consistency and robustness
over strategic agents with non-uniform weights. In particular, through a
reduction technique that identifies a subset of \emph{representative} instances
and maps the other given locations to the representative ones, we prove that
there exists a \emph{strategyproof} mechanism achieving a bounded consistency
guarantee of $\frac{\sqrt{(1+c)^2W^2_{\min}+(1-c)^2W^2_{\max}}}{(1+c)W_{\min}}$
and a bounded robustness guarantee of
$\frac{\sqrt{(1-c)^2W^2_{\min}+(1+c)^2W^2_{\max}}}{(1-c)W_{\min}}$ in weighted
settings, where $c$ can be viewed as a parameter to make a trade-off between
the consistency and robustness and $W_{\min}$ and $W_{\max}$ denote the minimum
and maximum agents' weight. We also proved that there is no strategyproof
deterministic mechanism that reach $1$-consistency and $O\left( n \cdot
\frac{W_{\max}}{W_{\min}} \right)$-robustness in weighted FLP, even with fully
predictions of all agents.

### Human-Computer Interaction

### 1. [Ragged Blocks: Rendering Structured Text with Style](http://arxiv.org/pdf/2507.06460v1)

Authors: Sam Cohen, Ravi Chugh

Whether it be source code in a programming language, prose in natural
language, or otherwise, text is highly structured. Currently, text
visualizations are confined either to _flat, line-based_ decorations, which can
convey only limited information about textual structure, or _nested boxes_,
which convey structure but often destroy the typographic layout of the
underlying text. We hypothesize that the lack of rich styling options limits
the kinds of information that are displayed alongside text, wherever it may be
displayed.
  In this paper, we show that it is possible to achieve arbitrarily nested
decorations while minimally disturbing the underlying typographic layout.
Specifically, we present a layout algorithm that generates _ragged blocks_, or
_rocks_, which are rectilinear polygons that allow nested text to be compactly
rendered even when styled with borders and padding.
  We evaluate our layout algorithm in two ways. First, on a benchmark suite
comprising representative source code files in multiple programming languages,
we show that the (ragged block) layouts produced by our algorithm are
substantially more compact than the (rectangular block) layouts produced by
conventional techniques, when uniformly styling every element in the syntax
tree with borders and padding. Second, through a small gallery of usage
scenarios, we demonstrate how future code editors, word processors, and other
document-rendering GUIs might convey rich semantic information through
domain-specific styling of ragged blocks.

### 2. [Smartphone Exergames with Real-Time Markerless Motion Capture: Challenges and Trade-offs](http://arxiv.org/pdf/2507.06669v1)

Authors: Mathieu Phosanarack, Laura Wallard, Sophie Lepreux, Christophe Kolski, Eugénie Avril

Markerless Motion Capture (MoCap) using smartphone cameras is a promising
approach to making exergames more accessible and cost-effective for health and
rehabilitation. Unlike traditional systems requiring specialized hardware,
recent advancements in AI-powered pose estimation enable movement tracking
using only a mobile device. For an upcoming study, a mobile application with
real-time exergames including markerless motion capture is being developed.
However, implementing such technology introduces key challenges, including
balancing accuracy and real-time responsiveness, ensuring proper user
interaction. Future research should explore optimizing AI models for realtime
performance, integrating adaptive gamification, and refining user-centered
design principles. By overcoming these challenges, smartphone-based exergames
could become powerful tools for engaging users in physical activity and
rehabilitation, extending their benefits to a broader audience.

### 3. [Effects of task difficulty and music expertise in virtual reality: Observations of cognitive load and task accuracy in a rhythm exergame](http://arxiv.org/pdf/2507.06691v1)

Authors: Kyla Ellahiyoun, Emma Jane Pretty, Renan Guarese, Marcel Takac, Haytham Fayek, Fabio Zambetta

This study explores the relationship between musical training, cognitive load
(CL), and task accuracy within the virtual reality (VR) exergame Beat Saber
across increasing levels of difficulty. Participants (N=32) completed a series
of post-task questionnaires after playing the game under three task difficulty
levels while having their physiological data measured by an Emotibit. Using
regression analyses, we found that task difficulty and gaming experience
significantly predicted subjective CL, whereas musical training did not.
However, musical training significantly predicted higher task accuracy, along
with lower subjective CL, increased gaming experience, and greater
physiological arousal. These results suggest that musical training enhances
task-specific performance but does not directly reduce subjective CL. Future
research should consider alternative methods of grouping musical expertise and
the additional predictability of flow and self-efficacy.

### 4. [Combining Human-centred Explainability and Explainable AI](http://arxiv.org/pdf/2507.06751v1)

Authors: Janin Koch, Vitor Fortes Rey

This position paper looks at differences between the current understandings
of human-centered explainability and explainability AI. We discuss current
ideas in both fields, as well as the differences and opportunities we
discovered. As an example of combining both, we will present preliminary work
on a new algebraic machine learning approach. We are excited to continue
discussing design opportunities for human-centered explainability (HCx) and xAI
with the broader HCxAI community.

### 5. [Toward Neurodivergent-Aware Productivity: A Systems and AI-Based Human-in-the-Loop Framework for ADHD-Affected Professionals](http://arxiv.org/pdf/2507.06864v1)

Authors: Raghavendra Deshmukh

Digital work environments in IT and knowledge-based sectors demand high
levels of attention management, task juggling, and self-regulation. For adults
with ADHD, these settings often amplify challenges such as time blindness,
digital distraction, emotional reactivity, and executive dysfunction. These
individuals prefer low-touch, easy-to-use interventions for daily tasks.
Conventional productivity tools often fail to support the cognitive variability
and overload experienced by neurodivergent professionals. This paper presents a
framework that blends Systems Thinking, Human-in-the-Loop design, AI/ML, and
privacy-first adaptive agents to support ADHD-affected users. The assistant
senses tab usage, application focus, and inactivity using on-device ML. These
cues are used to infer attention states and deliver nudges, reflective prompts,
or accountability-based presence (body doubling) that aid regulation without
disruption. Technically grounded in AI, the approach views attention as shaped
by dynamic feedback loops. The result is a replicable model for adaptive,
inclusive support tools in high-distraction work environments.

### 6. [Learning Japanese with Jouzu: Interaction Outcomes with Stylized Dialogue Fictional Agents](http://arxiv.org/pdf/2507.06483v1)

Authors: Zackary Rackauckas, Julia Hirschberg

This study investigates how stylized, voiced agents shape user interaction in
a multimodal language learning environment. We conducted a mixed-methods
evaluation of 54 participants interacting with anime-inspired characters
powered by large language models and expressive text-to-speech synthesis. These
agents responded in Japanese character language, offering users asynchronous,
semi-structured conversation in varying speech styles and emotional tones. We
analyzed user engagement patterns, perceived usability, emotional responses,
and learning behaviors, with particular attention to how agent stylization
influenced interaction across language proficiency levels and cultural
backgrounds. Our findings reveal that agent design, especially voice, persona,
and linguistic style, substantially affected user experience, motivation, and
strategy. This work contributes to the understanding of affective, culturally
stylized agents in human-agent interaction and offers guidance for designing
more engaging, socially responsive systems.

### 7. [Towards Designing Social Interventions for Online Climate Change Denialism Discussions](http://arxiv.org/pdf/2507.06561v1)

Authors: Ruican zhong, Shruti Phadke, Beth Goldberg, Tanushree Mitra

As conspiracy theories gain traction, it has become crucial to research
effective intervention strategies that can foster evidence and science-based
discussions in conspiracy theory communities online. This study presents a
novel framework using insider language to contest conspiracy theory ideology in
climate change denialism on Reddit. Focusing on discussions in two Reddit
communities, our research investigates reactions to pro-social and
evidence-based intervention messages for two cohorts of users: climate change
deniers and climate change supporters. Specifically, we combine manual and
generative AI-based methods to craft intervention messages and deploy the
interventions as replies on Reddit posts and comments through transparently
labeled bot accounts. On the one hand, we find that evidence-based
interventions with neutral language foster positive engagement, encouraging
open discussions among believers of climate change denialism. On the other,
climate change supporters respond positively, actively participating and
presenting additional evidence. Our study contributes valuable insights into
the process and challenges of automatically delivering interventions in
conspiracy theory communities on social media, and helps inform future research
on social media interventions.

### 8. [Integrating Perceptions: A Human-Centered Physical Safety Model for Human-Robot Interaction](http://arxiv.org/pdf/2507.06700v1)

Authors: Pranav Pandey, Ramviyas Parasuraman, Prashant Doshi

Ensuring safety in human-robot interaction (HRI) is essential to foster user
trust and enable the broader adoption of robotic systems. Traditional safety
models primarily rely on sensor-based measures, such as relative distance and
velocity, to assess physical safety. However, these models often fail to
capture subjective safety perceptions, which are shaped by individual traits
and contextual factors. In this paper, we introduce and analyze a parameterized
general safety model that bridges the gap between physical and perceived safety
by incorporating a personalization parameter, $\rho$, into the safety
measurement framework to account for individual differences in safety
perception. Through a series of hypothesis-driven human-subject studies in a
simulated rescue scenario, we investigate how emotional state, trust, and robot
behavior influence perceived safety. Our results show that $\rho$ effectively
captures meaningful individual differences, driven by affective responses,
trust in task consistency, and clustering into distinct user types.
Specifically, our findings confirm that predictable and consistent robot
behavior as well as the elicitation of positive emotional states, significantly
enhance perceived safety. Moreover, responses cluster into a small number of
user types, supporting adaptive personalization based on shared safety models.
Notably, participant role significantly shapes safety perception, and repeated
exposure reduces perceived safety for participants in the casualty role,
emphasizing the impact of physical interaction and experiential change. These
findings highlight the importance of adaptive, human-centered safety models
that integrate both psychological and behavioral dimensions, offering a pathway
toward more trustworthy and effective HRI in safety-critical domains.

### 9. [Tailoring deep learning for real-time brain-computer interfaces: From offline models to calibration-free online decoding](http://arxiv.org/pdf/2507.06779v1)

Authors: Martin Wimpff, Jan Zerfowski, Bin Yang

Despite the growing success of deep learning (DL) in offline brain-computer
interfaces (BCIs), its adoption in real-time applications remains limited due
to three primary challenges. First, most DL solutions are designed for offline
decoding, making the transition to online decoding unclear. Second, the use of
sliding windows in online decoding substantially increases computational
complexity. Third, DL models typically require large amounts of training data,
which are often scarce in BCI applications. To address these challenges and
enable real-time, cross-subject decoding without subject-specific calibration,
we introduce realtime adaptive pooling (RAP), a novel parameter-free method.
RAP seamlessly modifies the pooling layers of existing offline DL models to
meet online decoding requirements. It also reduces computational complexity
during training by jointly decoding consecutive sliding windows. To further
alleviate data requirements, our method leverages source-free domain
adaptation, enabling privacy-preserving adaptation across varying amounts of
target data. Our results demonstrate that RAP provides a robust and efficient
framework for real-time BCI applications. It preserves privacy, reduces
calibration demands, and supports co-adaptive BCI systems, paving the way for
broader adoption of DL in online BCIs. These findings lay a strong foundation
for developing user-centered, high-performance BCIs that facilitate immediate
feedback and user learning.

### 10. [Better frame rates or better visuals? An early report of Esports player practice in Dota 2](http://arxiv.org/pdf/2507.06790v1)

Authors: Arjun Madhusudan, Benjamin Watson

Esports athletes often reduce visual quality to improve latency and frame
rate, and increase their in-game performance. Little research has examined the
effects of this visuo-spatial tradeoff on performance, but we could find no
work studying how players manage this tradeoff in practice. This paper is an
initial examination of this question in the game Dota 2. First, we gather the
game configuration data of Dota 2 players in a small survey. We learn that
players do limit visual detail, particularly by turning off VSYNC, which
removes rendering/display synchronization delay but permits visual "tearing".
Second, we survey the intent of those same players with a few subjective
questions. Player intent matches configuration practice. While our sampling of
Dota 2 players may not be representative, our survey does reveal suggestive
trends that lay the groundwork for future, more rigorous and larger surveys.
Such surveys can help new players adapt to the game more quickly, encourage
researchers to investigate the relative importance of temporal and visual
detail, and justify design effort by developers in "low visual" game
configurations.

### Information Retrieval

### 1. [USD: A User-Intent-Driven Sampling and Dual-Debiasing Framework for Large-Scale Homepage Recommendations](http://arxiv.org/pdf/2507.06503v1)

Authors: Jiaqi Zheng, Cheng Guo, Yi Cao, Chaoqun Hou, Tong Liu, Bo Zheng

Large-scale homepage recommendations face critical challenges from
pseudo-negative samples caused by exposure bias, where non-clicks may indicate
inattention rather than disinterest. Existing work lacks thorough analysis of
invalid exposures and typically addresses isolated aspects (e.g., sampling
strategies), overlooking the critical impact of pseudo-positive samples - such
as homepage clicks merely to visit marketing portals. We propose a unified
framework for large-scale homepage recommendation sampling and debiasing. Our
framework consists of two key components: (1) a user intent-aware negative
sampling module to filter invalid exposure samples, and (2) an intent-driven
dual-debiasing module that jointly corrects exposure bias and click bias.
Extensive online experiments on Taobao demonstrate the efficacy of our
framework, achieving significant improvements in user click-through rates
(UCTR) by 35.4\% and 14.5\% in two variants of the marketing block on the
Taobao homepage, Baiyibutie and Taobaomiaosha.

### 2. [SPEAR: Subset-sampled Performance Evaluation via Automated Ground Truth Generation for RAG](http://arxiv.org/pdf/2507.06554v1)

Authors: Zou Yuheng, Wang Yiran, Tian Yuzhu, Zhu Min, Huang Yanhua

Retrieval-Augmented Generation (RAG) is a core approach for enhancing Large
Language Models (LLMs), where the effectiveness of the retriever largely
determines the overall response quality of RAG systems. Retrievers encompass a
multitude of hyperparameters that significantly impact performance outcomes and
demonstrate sensitivity to specific applications. Nevertheless, hyperparameter
optimization entails prohibitively high computational expenses. Existing
evaluation methods suffer from either prohibitive costs or disconnection from
domain-specific scenarios. This paper proposes SEARA (Subset sampling
Evaluation for Automatic Retriever Assessment), which addresses evaluation data
challenges through subset sampling techniques and achieves robust automated
retriever evaluation by minimal retrieval facts extraction and comprehensive
retrieval metrics. Based on real user queries, this method enables fully
automated retriever evaluation at low cost, thereby obtaining optimal retriever
for specific business scenarios. We validate our method across classic RAG
applications in rednote, including knowledge-based Q&A system and
retrieval-based travel assistant, successfully obtaining scenario-specific
optimal retrievers.

### 3. [Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study](http://arxiv.org/pdf/2507.06596v1)

Authors: Robin Ungruh, Alejandro Bellogín, Dominik Kowald, Maria Soledad Pera

Children are often exposed to items curated by recommendation algorithms.
Yet, research seldom considers children as a user group, and when it does, it
is anchored on datasets where children are underrepresented, risking
overlooking their interests, favoring those of the majority, i.e., mainstream
users. Recently, Ungruh et al. demonstrated that children's consumption
patterns and preferences differ from those of mainstream users, resulting in
inconsistent recommendation algorithm performance and behavior for this user
group. These findings, however, are based on two datasets with a limited child
user sample. We reproduce and replicate this study on a wider range of datasets
in the movie, music, and book domains, uncovering interaction patterns and
aspects of child-recommender interactions consistent across domains, as well as
those specific to some user samples in the data. We also extend insights from
the original study with popularity bias metrics, given the interpretation of
results from the original study. With this reproduction and extension, we
uncover consumption patterns and differences between age groups stemming from
intrinsic differences between children and others, and those unique to specific
datasets or domains.

### 4. [CDC: Causal Domain Clustering for Multi-Domain Recommendation](http://arxiv.org/pdf/2507.06877v1)

Authors: Huishi Luo, Yiqing Wu, Yiwen Chen, Fuzhen Zhuang, Deqing Wang

Multi-domain recommendation leverages domain-general knowledge to improve
recommendations across several domains. However, as platforms expand to dozens
or hundreds of scenarios, training all domains in a unified model leads to
performance degradation due to significant inter-domain differences. Existing
domain grouping methods, based on business logic or data similarities, often
fail to capture the true transfer relationships required for optimal grouping.
To effectively cluster domains, we propose Causal Domain Clustering (CDC). CDC
models domain transfer patterns within a large number of domains using two
distinct effects: the Isolated Domain Affinity Matrix for modeling
non-interactive domain transfers, and the Hybrid Domain Affinity Matrix for
considering dynamic domain synergy or interference under joint training. To
integrate these two transfer effects, we introduce causal discovery to
calculate a cohesion-based coefficient that adaptively balances their
contributions. A Co-Optimized Dynamic Clustering algorithm iteratively
optimizes target domain clustering and source domain selection for training.
CDC significantly enhances performance across over 50 domains on public
datasets and in industrial settings, achieving a 4.9% increase in online eCPM.
Code is available at
https://github.com/Chrissie-Law/Causal-Domain-Clustering-for-Multi-Domain-Recommendation

### 5. [Boosting Parameter Efficiency in LLM-Based Recommendation through Sophisticated Pruning](http://arxiv.org/pdf/2507.07064v1)

Authors: Shanle Zheng, Keqin Bao, Jizhi Zhang, Yang Zhang, Fuli Feng, Xiangnan He

LLM-based recommender systems have made significant progress; however, the
deployment cost associated with the large parameter volume of LLMs still
hinders their real-world applications. This work explores parameter pruning to
improve parameter efficiency while maintaining recommendation quality, thereby
enabling easier deployment. Unlike existing approaches that focus primarily on
inter-layer redundancy, we uncover intra-layer redundancy within components
such as self-attention and MLP modules. Building on this analysis, we propose a
more fine-grained pruning approach that integrates both intra-layer and
layer-wise pruning. Specifically, we introduce a three-stage pruning strategy
that progressively prunes parameters at different levels and parts of the
model, moving from intra-layer to layer-wise pruning, or from width to depth.
Each stage also includes a performance restoration step using distillation
techniques, helping to strike a balance between performance and parameter
efficiency. Empirical results demonstrate the effectiveness of our approach:
across three datasets, our models achieve an average of 88% of the original
model's performance while pruning more than 95% of the non-embedding
parameters. This underscores the potential of our method to significantly
reduce resource requirements without greatly compromising recommendation
quality. Our code will be available at: https://github.com/zheng-sl/PruneRec

### 6. [GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models](http://arxiv.org/pdf/2507.06507v1)

Authors: Zhen Yang, Haitao Lin, Jiawei xue, Ziji Zhang

In the past year, Generative Recommendations (GRs) have undergone substantial
advancements, especially in leveraging the powerful sequence modeling and
reasoning capabilities of Large Language Models (LLMs) to enhance overall
recommendation performance. LLM-based GRs are forming a new paradigm that is
distinctly different from discriminative recommendations, showing strong
potential to replace traditional recommendation systems heavily dependent on
complex hand-crafted features. In this paper, we provide a comprehensive survey
aimed at facilitating further research of LLM-based GRs. Initially, we outline
the general preliminaries and application cases of LLM-based GRs. Subsequently,
we introduce the main considerations when LLM-based GRs are applied in real
industrial scenarios. Finally, we explore promising directions for LLM-based
GRs. We hope that this survey contributes to the ongoing advancement of the GR
domain.

### 7. [DS@GT at CheckThat! 2025: Exploring Retrieval and Reranking Pipelines for Scientific Claim Source Retrieval on Social Media Discourse](http://arxiv.org/pdf/2507.06563v1)

Authors: Jeanette Schofield, Shuyu Tian, Hoang Thanh Thanh Truong, Maximilian Heil

Social media users often make scientific claims without citing where these
claims come from, generating a need to verify these claims. This paper details
work done by the DS@GT team for CLEF 2025 CheckThat! Lab Task 4b Scientific
Claim Source Retrieval which seeks to find relevant scientific papers based on
implicit references in tweets. Our team explored 6 different data augmentation
techniques, 7 different retrieval and reranking pipelines, and finetuned a
bi-encoder. Achieving an MRR@5 of 0.58, our team ranked 16th out of 30 teams
for the CLEF 2025 CheckThat! Lab Task 4b, and improvement of 0.15 over the BM25
baseline of 0.43. Our code is available on Github at
https://github.com/dsgt-arc/checkthat-2025-swd/tree/main/subtask-4b.

### 8. [Shifting from Ranking to Set Selection for Retrieval Augmented Generation](http://arxiv.org/pdf/2507.06838v1)

Authors: Dahyun Lee, Yongrae Jo, Haeju Park, Moontae Lee

Retrieval in Retrieval-Augmented Generation(RAG) must ensure that retrieved
passages are not only individually relevant but also collectively form a
comprehensive set. Existing approaches primarily rerank top-k passages based on
their individual relevance, often failing to meet the information needs of
complex queries in multi-hop question answering. In this work, we propose a
set-wise passage selection approach and introduce SETR, which explicitly
identifies the information requirements of a query through Chain-of-Thought
reasoning and selects an optimal set of passages that collectively satisfy
those requirements. Experiments on multi-hop RAG benchmarks show that SETR
outperforms both proprietary LLM-based rerankers and open-source baselines in
terms of answer correctness and retrieval quality, providing an effective and
efficient alternative to traditional rerankers in RAG systems. The code is
available at https://github.com/LGAI-Research/SetR

### 9. [MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval](http://arxiv.org/pdf/2507.06654v1)

Authors: Naoya Sogi, Takashi Shibata, Makoto Terao, Masanori Suganuma, Takayuki Okatani

Result diversification (RD) is a crucial technique in Text-to-Image Retrieval
for enhancing the efficiency of a practical application. Conventional methods
focus solely on increasing the diversity metric of image appearances. However,
the diversity metric and its desired value vary depending on the application,
which limits the applications of RD. This paper proposes a novel task called
CDR-CA (Contextual Diversity Refinement of Composite Attributes). CDR-CA aims
to refine the diversities of multiple attributes, according to the
application's context. To address this task, we propose Multi-Source DPPs, a
simple yet strong baseline that extends the Determinantal Point Process (DPP)
to multi-sources. We model MS-DPP as a single DPP model with a unified
similarity matrix based on a manifold representation. We also introduce Tangent
Normalization to reflect contexts. Extensive experiments demonstrate the
effectiveness of the proposed method. Our code is publicly available at
https://github.com/NEC-N-SOGI/msdpp.

### 10. [CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs](http://arxiv.org/pdf/2507.06715v1)

Authors: Garapati Keerthana, Manik Gupta

Large language models (LLMs), including zero-shot and few-shot paradigms,
have shown promising capabilities in clinical text generation. However,
real-world applications face two key challenges: (1) patient data is highly
unstructured, heterogeneous, and scattered across multiple note types and (2)
clinical notes are often long and semantically dense, making naive prompting
infeasible due to context length constraints and the risk of omitting
clinically relevant information.
  We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a
domain-specific framework for structured and clinically grounded text
generation using LLMs. It incorporates a novel hierarchical chunking strategy
that respects clinical document structure and introduces a task-specific
dual-stage retrieval mechanism. The global stage identifies relevant note types
using evidence-based queries, while the local stage extracts high-value content
within those notes creating relevance at both document and section levels.
  We apply the system to generate structured progress notes for individual
hospital visits using 15 clinical note types from the MIMIC-III dataset.
Experiments show that it preserves temporal and semantic alignment across
visits, achieving an average alignment score of 87.7%, surpassing the 80.7%
baseline from real clinician-authored notes. The generated outputs also
demonstrate high consistency across LLMs, reinforcing deterministic behavior
essential for reproducibility, reliability, and clinical trust.

### Machine Learning

### 1. [FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](http://arxiv.org/pdf/2507.06482v1)

Authors: Huan Wang, Haoran Li, Huaming Chen, Jun Yan, Jiahua Shi, Jun Shen

Federated learning aims at training models collaboratively across
participants while protecting privacy. However, one major challenge for this
paradigm is the data heterogeneity issue, where biased data preferences across
multiple clients, harming the model's convergence and performance. In this
paper, we first introduce powerful diffusion models into the federated learning
paradigm and show that diffusion representations are effective steers during
federated training. To explore the possibility of using diffusion
representations in handling data heterogeneity, we propose a novel
diffusion-inspired Federated paradigm with Diffusion Representation
Collaboration, termed FedDifRC, leveraging meaningful guidance of diffusion
models to mitigate data heterogeneity. The key idea is to construct text-driven
diffusion contrasting and noise-driven diffusion regularization, aiming to
provide abundant class-related semantic information and consistent convergence
signals. On the one hand, we exploit the conditional feedback from the
diffusion model for different text prompts to build a text-driven contrastive
learning strategy. On the other hand, we introduce a noise-driven consistency
regularization to align local instances with diffusion denoising
representations, constraining the optimization region in the feature space. In
addition, FedDifRC can be extended to a self-supervised scheme without relying
on any labeled data. We also provide a theoretical analysis for FedDifRC to
ensure convergence under non-convex objectives. The experiments on different
scenarios validate the effectiveness of FedDifRC and the efficiency of crucial
components.

### 2. [Direct Regret Optimization in Bayesian Optimization](http://arxiv.org/pdf/2507.06529v1)

Authors: Fengxue Zhang, Yuxin Chen

Bayesian optimization (BO) is a powerful paradigm for optimizing expensive
black-box functions. Traditional BO methods typically rely on separate
hand-crafted acquisition functions and surrogate models for the underlying
function, and often operate in a myopic manner. In this paper, we propose a
novel direct regret optimization approach that jointly learns the optimal model
and non-myopic acquisition by distilling from a set of candidate models and
acquisitions, and explicitly targets minimizing the multi-step regret. Our
framework leverages an ensemble of Gaussian Processes (GPs) with varying
hyperparameters to generate simulated BO trajectories, each guided by an
acquisition function chosen from a pool of conventional choices, until a
Bayesian early stop criterion is met. These simulated trajectories, capturing
multi-step exploration strategies, are used to train an end-to-end decision
transformer that directly learns to select next query points aimed at improving
the ultimate objective. We further adopt a dense training--sparse learning
paradigm: The decision transformer is trained offline with abundant simulated
data sampled from ensemble GPs and acquisitions, while a limited number of real
evaluations refine the GPs online. Experimental results on synthetic and
real-world benchmarks suggest that our method consistently outperforms BO
baselines, achieving lower simple regret and demonstrating more robust
exploration in high-dimensional or noisy settings.

### 3. [Generalization in Reinforcement Learning for Radio Access Networks](http://arxiv.org/pdf/2507.06602v1)

Authors: Burak Demirel, Yu Wang, Cristian Tatino, Pablo Soldati

Modern RAN operate in highly dynamic and heterogeneous environments, where
hand-tuned, rule-based RRM algorithms often underperform. While RL can surpass
such heuristics in constrained settings, the diversity of deployments and
unpredictable radio conditions introduce major generalization challenges.
Data-driven policies frequently overfit to training conditions, degrading
performance in unseen scenarios. To address this, we propose a
generalization-centered RL framework for RAN control that: (i) encodes cell
topology and node attributes via attention-based graph representations; (ii)
applies domain randomization to broaden the training distribution; and (iii)
distributes data generation across multiple actors while centralizing training
in a cloud-compatible architecture aligned with O-RAN principles. Although
generalization increases computational and data-management complexity, our
distributed design mitigates this by scaling data collection and training
across diverse network conditions. Applied to downlink link adaptation in five
5G benchmarks, our policy improves average throughput and spectral efficiency
by ~10% over an OLLA baseline (10% BLER target) in full-buffer MIMO/mMIMO and
by >20% under high mobility. It matches specialized RL in full-buffer traffic
and achieves up to 4- and 2-fold gains in eMBB and mixed-traffic benchmarks,
respectively. In nine-cell deployments, GAT models offer 30% higher throughput
over MLP baselines. These results, combined with our scalable architecture,
offer a path toward AI-native 6G RAN using a single, generalizable RL agent.

### 4. [UniOD: A Universal Model for Outlier Detection across Diverse Domains](http://arxiv.org/pdf/2507.06624v1)

Authors: Dazhi Fu, Jicong Fan

Outlier detection (OD) seeks to distinguish inliers and outliers in
completely unlabeled datasets and plays a vital role in science and
engineering. Most existing OD methods require troublesome dataset-specific
hyperparameter tuning and costly model training before they can be deployed to
identify outliers. In this work, we propose UniOD, a universal OD framework
that leverages labeled datasets to train a single model capable of detecting
outliers of datasets from diverse domains. Specifically, UniOD converts each
dataset into multiple graphs, produces consistent node features, and frames
outlier detection as a node-classification task, and is able to generalize to
unseen domains. As a result, UniOD avoids effort on model selection and
hyperparameter tuning, reduces computational cost, and effectively utilizes the
knowledge from historical datasets, which improves the convenience and accuracy
in real applications. We evaluate UniOD on 15 benchmark OD datasets against 15
state-of-the-art baselines, demonstrating its effectiveness.

### 5. [Prevention of Overfitting on Mesh-Structured Data Regressions with a Modified Laplace Operator](http://arxiv.org/pdf/2507.06631v1)

Authors: Enda D. V. Bigarella

This document reports on a method for detecting and preventing overfitting on
data regressions, herein applied to mesh-like data structures. The mesh
structure allows for the straightforward computation of the Laplace-operator
second-order derivatives in a finite-difference fashion for noiseless data.
Derivatives of the training data are computed on the original training mesh to
serve as a true label of the entropy of the training data. Derivatives of the
trained data are computed on a staggered mesh to identify oscillations in the
interior of the original training mesh cells. The loss of the Laplace-operator
derivatives is used for hyperparameter optimisation, achieving a reduction of
unwanted oscillation through the minimisation of the entropy of the trained
model. In this setup, testing does not require the splitting of points from the
training data, and training is thus directly performed on all available
training points. The Laplace operator applied to the trained data on a
staggered mesh serves as a surrogate testing metric based on diffusion
properties.

### 6. [Federated Learning Inspired Fuzzy Systems: Decentralized Rule Updating for Privacy and Scalable Decision Making](http://arxiv.org/pdf/2507.06652v1)

Authors: Arthur Alexander Lim, Zhen Bin It, Jovan Bowen Heng, Tee Hui Teo

Fuzzy systems are a way to allow machines, systems and frameworks to deal
with uncertainty, which is not possible in binary systems that most computers
use. These systems have already been deployed for certain use cases, and fuzzy
systems could be further improved as proposed in this paper. Such technologies
to draw inspiration from include machine learning and federated learning.
Machine learning is one of the recent breakthroughs of technology and could be
applied to fuzzy systems to further improve the results it produces. Federated
learning is also one of the recent technologies that have huge potential, which
allows machine learning training to improve by reducing privacy risk, reducing
burden on networking infrastructure, and reducing latency of the latest model.
Aspects from federated learning could be used to improve federated learning,
such as applying the idea of updating the fuzzy rules that make up a key part
of fuzzy systems, to further improve it over time. This paper discusses how
these improvements would be implemented in fuzzy systems, and how it would
improve fuzzy systems. It also discusses certain limitations on the potential
improvements. It concludes that these proposed ideas and improvements require
further investigation to see how far the improvements are, but the potential is
there to improve fuzzy systems.

### 7. [Value from Observations: Towards Large-Scale Imitation Learning via Self-Improvement](http://arxiv.org/pdf/2507.06701v1)

Authors: Michael Bloesch, Markus Wulfmeier, Philemon Brakel, Todor Davchev, Martina Zambelli, Jost Tobias Springenberg, Abbas Abdolmaleki, William F Whitney, Nicolas Heess, Roland Hafner, Martin Riedmiller

Imitation Learning from Observation (IfO) offers a powerful way to learn
behaviors at large-scale: Unlike behavior cloning or offline reinforcement
learning, IfO can leverage action-free demonstrations and thus circumvents the
need for costly action-labeled demonstrations or reward functions. However,
current IfO research focuses on idealized scenarios with mostly bimodal-quality
data distributions, restricting the meaningfulness of the results. In contrast,
this paper investigates more nuanced distributions and introduces a method to
learn from such data, moving closer to a paradigm in which imitation learning
can be performed iteratively via self-improvement. Our method adapts RL-based
imitation learning to action-free demonstrations, using a value function to
transfer information between expert and non-expert data. Through comprehensive
evaluation, we delineate the relation between different data distributions and
the applicability of algorithms and highlight the limitations of established
methods. Our findings provide valuable insights for developing more robust and
practical IfO techniques on a path to scalable behaviour learning.

### 8. [Robust Deep Network Learning of Nonlinear Regression Tasks by Parametric Leaky Exponential Linear Units (LELUs) and a Diffusion Metric](http://arxiv.org/pdf/2507.06765v1)

Authors: Enda D. V. Bigarella

This document proposes a parametric activation function (ac.f.) aimed at
improving multidimensional nonlinear data regression. It is a established
knowledge that nonlinear ac.f.'s are required for learning nonlinear datasets.
This work shows that smoothness and gradient properties of the ac.f. further
impact the performance of large neural networks in terms of overfitting and
sensitivity to model parameters. Smooth but vanishing-gradient ac.f.'s such as
ELU or SiLU have limited performance and non-smooth ac.f.'s such as RELU and
Leaky-RELU further impart discontinuity in the trained model. Improved
performance is demonstrated with a smooth "Leaky Exponential Linear Unit", with
non-zero gradient that can be trained. A novel diffusion-loss metric is also
proposed to gauge the performance of the trained models in terms of
overfitting.

### 9. [Speech Tokenizer is Key to Consistent Representation](http://arxiv.org/pdf/2507.06802v1)

Authors: Wonjin Jung, Sungil Kang, Dong-Yeon Cho

Speech tokenization is crucial in digital speech processing, converting
continuous speech signals into discrete units for various computational tasks.
This paper introduces a novel speech tokenizer with broad applicability across
downstream tasks. While recent advances in residual vector quantization (RVQ)
have incorporated semantic elements, they often neglect critical acoustic
features. We propose an advanced approach that simultaneously encodes both
linguistic and acoustic information, preserving prosodic and emotional content.
Our method significantly enhances speech representation fidelity across diverse
applications. Empirical evaluations demonstrate its effectiveness in speech
coding, voice conversion, emotion recognition, and multimodal language
modeling, without requiring additional training. This versatility underscores
its potential as a key tool for advancing AI-driven speech processing.

### 10. [Episodic Contextual Bandits with Knapsacks under Conversion Models](http://arxiv.org/pdf/2507.06859v1)

Authors: Zitian Li, Wang Chi Cheung

We study an online setting, where a decision maker (DM) interacts with
contextual bandit-with-knapsack (BwK) instances in repeated episodes. These
episodes start with different resource amounts, and the contexts' probability
distributions are non-stationary in an episode. All episodes share the same
latent conversion model, which governs the random outcome contingent upon a
request's context and an allocation decision. Our model captures applications
such as dynamic pricing on perishable resources with episodic replenishment,
and first price auctions in repeated episodes with different starting budgets.
We design an online algorithm that achieves a regret sub-linear in $T$, the
number of episodes, assuming access to a \emph{confidence bound oracle} that
achieves an $o(T)$-regret. Such an oracle is readily available from existing
contextual bandit literature. We overcome the technical challenge with
arbitrarily many possible contexts, which leads to a reinforcement learning
problem with an unbounded state space. Our framework provides improved regret
bounds in certain settings when the DM is provided with unlabeled feature data,
which is novel to the contextual BwK literature.

### Neural and Evolutionary Computing

### 1. [Energy-Efficient Supervised Learning with a Binary Stochastic Forward-Forward Algorithm](http://arxiv.org/pdf/2507.06461v1)

Authors: Risi Jaiswal, Supriyo Datta, Joseph G. Makin

Reducing energy consumption has become a pressing need for modern machine
learning, which has achieved many of its most impressive results by scaling to
larger and more energy-consumptive neural networks. Unfortunately, the main
algorithm for training such networks, backpropagation, poses significant
challenges for custom hardware accelerators, due to both its serial
dependencies and the memory footprint needed to store forward activations for
the backward pass. Alternatives to backprop, although less effective, do exist;
here the main computational bottleneck becomes matrix multiplication. In this
study, we derive forward-forward algorithms for binary, stochastic units.
Binarization of the activations transforms matrix multiplications into indexing
operations, which can be executed efficiently in hardware. Stochasticity,
combined with tied weights across units with different biases, bypasses the
information bottleneck imposed by binary units. Furthermore, although slow and
expensive in traditional hardware, binary sampling that is very fast can be
implemented cheaply with p-bits (probabilistic bits), novel devices made up of
unstable magnets. We evaluate our proposed algorithms on the MNIST,
Fashion-MNIST, and CIFAR-10 datasets, showing that its performance is close to
real-valued forward-forward, but with an estimated energy savings of about one
order of magnitude.

### 2. [Designing Robust Software Sensors for Nonlinear Systems via Neural Networks and Adaptive Sliding Mode Control](http://arxiv.org/pdf/2507.06817v1)

Authors: Ayoub Farkane, Mohamed Boutayeb, Mustapha Oudani, Mounir Ghogho

Accurate knowledge of the state variables in a dynamical system is critical
for effective control, diagnosis, and supervision, especially when direct
measurements of all states are infeasible. This paper presents a novel approach
to designing software sensors for nonlinear dynamical systems expressed in
their most general form. Unlike traditional model-based observers that rely on
explicit transformations or linearization, the proposed framework integrates
neural networks with adaptive Sliding Mode Control (SMC) to design a robust
state observer under a less restrictive set of conditions. The learning process
is driven by available sensor measurements, which are used to correct the
observer's state estimate. The training methodology leverages the system's
governing equations as a physics-based constraint, enabling observer synthesis
without access to ground-truth state trajectories. By employing a time-varying
gain matrix dynamically adjusted by the neural network, the observer adapts in
real-time to system changes, ensuring robustness against noise, external
disturbances, and variations in system dynamics. Furthermore, we provide
sufficient conditions to guarantee estimation error convergence, establishing a
theoretical foundation for the observer's reliability. The methodology's
effectiveness is validated through simulations on challenging examples,
including systems with non-differentiable dynamics and varying observability
conditions. These examples, which are often problematic for conventional
techniques, serve to demonstrate the robustness and broad applicability of our
approach. The results show rapid convergence and high accuracy, underscoring
the method's potential for addressing complex state estimation challenges in
real-world applications.

### Networking and Internet Architecture

### 1. [Stacked Intelligent Metasurfaces-Aided eVTOL Delay Sensitive Communications](http://arxiv.org/pdf/2507.06632v1)

Authors: Liyuan Chen, Kai Xiong, Yujie Qin, Hanqing Yu, Supeng Leng, Chau Yuen

With rapid urbanization and increasing population density, urban traffic
congestion has become a critical issue, and traditional ground transportation
methods are no longer sufficient to address it effectively. To tackle this
challenge, the concept of Advanced Air Mobility (AAM) has emerged, aiming to
utilize low-altitude airspace to establish a three-dimensional transportation
system. Among various components of the AAM system, electric vertical take-off
and landing (eVTOL) aircraft plays a pivotal role due to their flexibility and
efficiency. However, the immaturity of Ultra Reliable Low Latency Communication
(URLLC) technologies poses significant challenges to safety-critical AAM
operations. Specifically, existing Stacked Intelligent Metasurfaces (SIM)-based
eVTOL systems lack rigorous mathematical frameworks to quantify probabilistic
delay bounds under dynamic air traffic patterns, a prerequisite for collision
avoidance and airspace management. To bridge this gap, we employ network
calculus tools to derive the probabilistic upper bound on communication delay
in the AAM system for the first time. Furthermore, we formulate a complex
non-convex optimization problem that jointly minimizes the probabilistic delay
bound and the propagation delay. To solve this problem efficiently, we propose
a solution based on the Block Coordinate Descent (BCD) algorithm and
Semidefinite Relaxation (SDR) method. In addition, we conduct a comprehensive
analysis of how various factors impact regret and transmission rate, and
explore the influence of varying load intensity and total delay on the
probabilistic delay bound.

### 2. [Learning To Communicate Over An Unknown Shared Network](http://arxiv.org/pdf/2507.06499v1)

Authors: Shivangi Agarwal, Adi Asija, Sanjit K. Kaul, Arani Bhattacharya, Saket Anand

As robots (edge-devices, agents) find uses in an increasing number of
settings and edge-cloud resources become pervasive, wireless networks will
often be shared by flows of data traffic that result from communication between
agents and corresponding edge-cloud. In such settings, agent communicating with
the edge-cloud is unaware of state of network resource, which evolves in
response to not just agent's own communication at any given time but also to
communication by other agents, which stays unknown to the agent. We address
challenge of an agent learning a policy that allows it to decide whether or not
to communicate with its cloud node, using limited feedback it obtains from its
own attempts to communicate, to optimize its utility. The policy generalizes
well to any number of other agents sharing the network and must not be trained
for any particular network configuration. Our proposed policy is a DRL model
Query Net (QNet) that we train using a proposed simulation-to-real framework.
Our simulation model has just one parameter and is agnostic to specific
configurations of any wireless network. It allows training an agent's policy
over a wide range of outcomes that an agent's communication with its edge-cloud
node may face when using a shared network, by suitably randomizing the
simulation parameter. We propose a learning algorithm that addresses challenges
observed in training QNet. We validate our simulation-to-real driven approach
through experiments conducted on real wireless networks including WiFi and
cellular. We compare QNet with other policies to demonstrate its efficacy. WiFi
experiments involved as few as five agents, resulting in barely any contention
for the network, to as many as fifty agents, resulting in severe contention.
The cellular experiments spanned a broad range of network conditions, with
baseline RTT ranging from a low of 0.07 second to a high of 0.83 second.

### 3. [Connecting the Unconnected -- Sentiment Analysis of Field Survey of Internet Connectivity in Emerging Economies](http://arxiv.org/pdf/2507.06827v1)

Authors: Dibakar Das, Barath S Narayan, Aarna Bhammar, Jyotsna Bapat

Internet has significantly improved the quality of citizens across the world.
Though the internet coverage is quite high, 40% of global population do not
have access to broadband internet. This paper presents an analysis of a field
survey of population in some areas of Kathmandu, Nepal, an emerging economy.
This survey was triggered by intermittent severe congestion of internet in
certain areas of the city. People from three different areas were asked about
their present experience of internet usage, its impact on their lives and their
aspirations for the future. Survey pointed to high speed, low cost, reliable
and secure internet as a major aspiration of the respondents. Based on their
inputs, this paper presents a sentiment analysis as well as demographic
information. Keys insights from this analysis shows that overall sentiment to
most queries are positive. The variances of positive sentiments are high
whereas those for negative ones are low. Also, some correlations and clusters
are observed among the attributes though no dominant component exists in the
data.

### 4. [SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference](http://arxiv.org/pdf/2507.06567v1)

Authors: Qian Chen, Xianhao Chen, Kaibin Huang

Mixture-of-Experts (MoE) models improve the scalability of large language
models (LLMs) by activating only a small subset of relevant experts per input.
However, the sheer number of expert networks in an MoE model introduces a
significant storage burden for an edge device. To address this challenge, we
consider a scenario where experts are dispersed within an edge network for
distributed inference. Based on the popular Top-$K$ expert selection strategy,
we formulate a latency minimization problem by optimizing expert caching on
edge servers under storage constraints. When $K=1$, the problem reduces to a
monotone submodular maximization problem with knapsack constraints, for which
we design a greedy-based algorithm with a $(1 - 1/e)$-approximation guarantee.
For the general case where $K\geq1$, expert co-activation within the same MoE
layer introduces non-submodularity, causing greedy methods to be ineffective.
To tackle this issue, we propose a successive greedy decomposition method to
decompose the original problem into a series of subproblems, with each being
solved by a dynamic programming approach. Furthermore, we design an accelerated
algorithm based on the max-convolution technique to obtain the approximate
solution with a provable guarantee in polynomial time. Simulation results on
various MoE models demonstrate that our method significantly reduces inference
latency compared to existing baselines.

### 5. [Beyond Connectivity: An Open Architecture for AI-RAN Convergence in 6G](http://arxiv.org/pdf/2507.06911v1)

Authors: Michele Polese, Niloofar Mohamadi, Salvatore D'Oro, Tommaso Melodia

The proliferation of data-intensive Artificial Intelligence (AI) applications
at the network edge demands a fundamental shift in RAN design, from merely
consuming AI for network optimization, to actively enabling distributed AI
workloads. This paradigm shift presents a significant opportunity for network
operators to monetize AI at the edge while leveraging existing infrastructure
investments. To realize this vision, this article presents a novel converged
O-RAN and AI-RAN architecture that unifies orchestration and management of both
telecommunications and AI workloads on shared infrastructure. The proposed
architecture extends the Open RAN principles of modularity, disaggregation, and
cloud-nativeness to support heterogeneous AI deployments. We introduce two key
architectural innovations: (i) the AI-RAN Orchestrator, which extends the O-RAN
Service Management and Orchestration (SMO) to enable integrated resource and
allocation across RAN and AI workloads; and (ii) AI-RAN sites that provide
distributed edge AI platforms with real-time processing capabilities. The
proposed system supports flexible deployment options, allowing AI workloads to
be orchestrated with specific timing requirements (real-time or batch
processing) and geographic targeting. The proposed architecture addresses the
orchestration requirements for managing heterogeneous workloads at different
time scales while maintaining open, standardized interfaces and multi-vendor
interoperability.

### 6. [Optimizing Cognitive Networks: Reinforcement Learning Meets Energy Harvesting Over Cascaded Channels](http://arxiv.org/pdf/2507.06981v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

This paper presents a reinforcement learning (RL) based approach to improve
the physical layer security (PLS) of an underlay cognitive radio network (CRN)
over cascaded channels. These channels are utilized in highly mobile networks
such as cognitive vehicular networks (CVN). In addition, an eavesdropper aims
to intercept the communications between secondary users (SUs). The SU receiver
has full-duplex and energy harvesting capabilities to generate jamming signals
to confound the eavesdropper and enhance security. Moreover, the SU transmitter
extracts energy from ambient radio frequency signals in order to power
subsequent transmissions to its intended receiver. To optimize the privacy and
reliability of the SUs in a CVN, a deep Q-network (DQN) strategy is utilized
where multiple DQN agents are required such that an agent is assigned at each
SU transmitter. The objective for the SUs is to determine the optimal
transmission power and decide whether to collect energy or transmit messages
during each time period in order to maximize their secrecy rate. Thereafter, we
propose a DQN approach to maximize the throughput of the SUs while respecting
the interference threshold acceptable at the receiver of the primary user.
According to our findings, our strategy outperforms two other baseline
strategies in terms of security and reliability.

### 7. [Maximizing Reliability in Overlay Radio Networks with Time Switching and Power Splitting Energy Harvesting](http://arxiv.org/pdf/2507.06983v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

Cognitive radio networks (CRNs) are acknowledged for their ability to tackle
the issue of spectrum under-utilization. In the realm of CRNs, this paper
investigates the energy efficiency issue and addresses the critical challenge
of optimizing system reliability for overlay CRN access mode. Randomly
dispersed secondary users (SUs) serving as relays for primary users (PUs) are
considered, in which one of these relays is designated to harvest energy
through the time switching-energy harvesting (EH) protocol. Moreover, this
relay amplifies-and-forwards (AF) the PU's messages and broadcasts them along
with its own across cascaded $\kappa$-$\mu$ fading channels. The power
splitting protocol is another EH approach utilized by the SU and PU receivers
to enhance the amount of energy in their storage devices. In addition, the SU
transmitters and the SU receiver are deployed with multiple antennas for
reception and apply the maximal ratio combining approach. The outage
probability is utilized to assess both networks' reliability. Then, an energy
efficiency evaluation is performed to determine the effectiveness of EH on the
system. Finally, an optimization problem is provided with the goal of
maximizing the data rate of the SUs by optimizing the time switching and the
power allocation parameters of the SU relay.

### 8. [Federated Learning-based MARL for Strengthening Physical-Layer Security in B5G Networks](http://arxiv.org/pdf/2507.06997v1)

Authors: Deemah H. Tashman, Soumaya Cherkaoui, Walaa Hamouda

This paper explores the application of a federated learning-based multi-agent
reinforcement learning (MARL) strategy to enhance physical-layer security (PLS)
in a multi-cellular network within the context of beyond 5G networks. At each
cell, a base station (BS) operates as a deep reinforcement learning (DRL) agent
that interacts with the surrounding environment to maximize the secrecy rate of
legitimate users in the presence of an eavesdropper. This eavesdropper attempts
to intercept the confidential information shared between the BS and its
authorized users. The DRL agents are deemed to be federated since they only
share their network parameters with a central server and not the private data
of their legitimate users. Two DRL approaches, deep Q-network (DQN) and
Reinforce deep policy gradient (RDPG), are explored and compared. The results
demonstrate that RDPG converges more rapidly than DQN. In addition, we
demonstrate that the proposed method outperforms the distributed DRL approach.
Furthermore, the outcomes illustrate the trade-off between security and
complexity.

### Robotics

### 1. [KLEIYN : A Quadruped Robot with an Active Waist for Both Locomotion and Wall Climbing](http://arxiv.org/pdf/2507.06562v1)

Authors: Keita Yoneda, Kento Kawaharazuka, Temma Suzuki, Takahiro Hattori, Kei Okada

In recent years, advancements in hardware have enabled quadruped robots to
operate with high power and speed, while robust locomotion control using
reinforcement learning (RL) has also been realized. As a result, expectations
are rising for the automation of tasks such as material transport and
exploration in unknown environments. However, autonomous locomotion in rough
terrains with significant height variations requires vertical movement, and
robots capable of performing such movements stably, along with their control
methods, have not yet been fully established. In this study, we developed the
quadruped robot KLEIYN, which features a waist joint, and aimed to expand
quadruped locomotion by enabling chimney climbing through RL. To facilitate the
learning of vertical motion, we introduced Contact-Guided Curriculum Learning
(CGCL). As a result, KLEIYN successfully climbed walls ranging from 800 mm to
1000 mm in width at an average speed of 150 mm/s, 50 times faster than
conventional robots. Furthermore, we demonstrated that the introduction of a
waist joint improves climbing performance, particularly enhancing tracking
ability on narrow walls.

### 2. [AI Space Cortex: An Experimental System for Future Era Space Exploration](http://arxiv.org/pdf/2507.06574v1)

Authors: Thomas Touma, Ersin Daş, Erica Tevere, Martin Feather, Ksenia Kolcio, Maurice Prather, Alberto Candela, Ashish Goel, Erik Kramer, Hari Nayar, Lorraine Fesq, Joel W. Burdick

Our Robust, Explainable Autonomy for Scientific Icy Moon Operations (REASIMO)
effort contributes to NASA's Concepts for Ocean worlds Life Detection
Technology (COLDTech) program, which explores science platform technologies for
ocean worlds such as Europa and Enceladus. Ocean world missions pose
significant operational challenges. These include long communication lags,
limited power, and lifetime limitations caused by radiation damage and hostile
conditions. Given these operational limitations, onboard autonomy will be vital
for future Ocean world missions. Besides the management of nominal lander
operations, onboard autonomy must react appropriately in the event of
anomalies. Traditional spacecraft rely on a transition into 'safe-mode' in
which non-essential components and subsystems are powered off to preserve
safety and maintain communication with Earth. For a severely time-limited Ocean
world mission, resolutions to these anomalies that can be executed without
Earth-in-the-loop communication and associated delays are paramount for
completion of the mission objectives and science goals. To address these
challenges, the REASIMO effort aims to demonstrate a robust level of
AI-assisted autonomy for such missions, including the ability to detect and
recover from anomalies, and to perform missions based on pre-trained behaviors
rather than hard-coded, predetermined logic like all prior space missions. We
developed an AI-assisted, personality-driven, intelligent framework for control
of an Ocean world mission by combining a mix of advanced technologies. To
demonstrate the capabilities of the framework, we perform tests of autonomous
sampling operations on a lander-manipulator testbed at the NASA Jet Propulsion
Laboratory, approximating possible surface conditions such a mission might
encounter.

### 3. [Growing Trees with an Agent: Accelerating RRTs with Learned, Multi-Step Episodic Exploration](http://arxiv.org/pdf/2507.06605v1)

Authors: Xinyu Wu

Classical sampling-based motion planners like the RRTs suffer from
inefficiencies, particularly in cluttered or high-dimensional spaces, due to
their reliance on undirected, random sampling. This paper introduces the
Episodic RRT, a novel hybrid planning framework that replaces the primitive of
a random point with a learned, multi-step "exploratory episode" generated by a
Deep Reinforcement Learning agent. By making the DRL agent the engine of
exploration, ERRT transforms the search process from a diffuse, volumetric
expansion into a directed, branch-like growth. This paradigm shift yields key
advantages: it counters the curse of dimensionality with focused exploration,
minimizes expensive collision checks by proactively proposing locally valid
paths, and improves connectivity by generating inherently connected path
segments. We demonstrate through extensive empirical evaluation across 2D, 3D,
and 6D environments that ERRT and its variants consistently and significantly
outperform their classical counterparts. In a challenging 6D robotic arm
scenario, ERRT achieves a 98% success rate compared to 19% for RRT, is up to
107x faster, reduces collision checks by over 99.6%, and finds initial paths
that are nearly 50% shorter. Furthermore, its asymptotically optimal variant,
ERRT*, demonstrates vastly superior anytime performance, refining solutions to
near-optimality up to 29x faster than standard RRT* in 3D environments. Code:
https://xinyuwuu.github.io/Episodic_RRT/.

### 4. [Multi-Task Multi-Agent Reinforcement Learning via Skill Graphs](http://arxiv.org/pdf/2507.06690v1)

Authors: Guobin Zhu, Rui Zhou, Wenkang Ji, Hongyin Zhang, Donglin Wang, Shiyu Zhao

Multi-task multi-agent reinforcement learning (MT-MARL) has recently gained
attention for its potential to enhance MARL's adaptability across multiple
tasks. However, it is challenging for existing multi-task learning methods to
handle complex problems, as they are unable to handle unrelated tasks and
possess limited knowledge transfer capabilities. In this paper, we propose a
hierarchical approach that efficiently addresses these challenges. The
high-level module utilizes a skill graph, while the low-level module employs a
standard MARL algorithm. Our approach offers two contributions. First, we
consider the MT-MARL problem in the context of unrelated tasks, expanding the
scope of MTRL. Second, the skill graph is used as the upper layer of the
standard hierarchical approach, with training independent of the lower layer,
effectively handling unrelated tasks and enhancing knowledge transfer
capabilities. Extensive experiments are conducted to validate these advantages
and demonstrate that the proposed method outperforms the latest hierarchical
MAPPO algorithms. Videos and code are available at
https://github.com/WindyLab/MT-MARL-SG

### 5. [Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](http://arxiv.org/pdf/2507.06710v1)

Authors: Zhenyang Liu, Yikai Wang, Kuanning Wang, Longfei Liang, Xiangyang Xue, Yanwei Fu

Visual imitation learning is effective for robots to learn versatile tasks.
However, many existing methods rely on behavior cloning with supervised
historical trajectories, limiting their 3D spatial and 4D spatiotemporal
awareness. Consequently, these methods struggle to capture the 3D structures
and 4D spatiotemporal relationships necessary for real-world deployment. In
this work, we propose 4D Diffusion Policy (DP4), a novel visual imitation
learning method that incorporates spatiotemporal awareness into diffusion-based
policies. Unlike traditional approaches that rely on trajectory cloning, DP4
leverages a dynamic Gaussian world model to guide the learning of 3D spatial
and 4D spatiotemporal perceptions from interactive environments. Our method
constructs the current 3D scene from a single-view RGB-D observation and
predicts the future 3D scene, optimizing trajectory generation by explicitly
modeling both spatial and temporal dependencies. Extensive experiments across
17 simulation tasks with 173 variants and 3 real-world robotic tasks
demonstrate that the 4D Diffusion Policy (DP4) outperforms baseline methods,
improving the average simulation task success rate by 16.4% (Adroit), 14%
(DexArt), and 6.45% (RLBench), and the average real-world robotic task success
rate by 8.6%.

### 6. [Hierarchical Reinforcement Learning for Articulated Tool Manipulation with Multifingered Hand](http://arxiv.org/pdf/2507.06822v1)

Authors: Wei Xu, Yanchao Zhao, Weichao Guo, Xinjun Sheng

Manipulating articulated tools, such as tweezers or scissors, has rarely been
explored in previous research. Unlike rigid tools, articulated tools change
their shape dynamically, creating unique challenges for dexterous robotic
hands. In this work, we present a hierarchical, goal-conditioned reinforcement
learning (GCRL) framework to improve the manipulation capabilities of
anthropomorphic robotic hands using articulated tools. Our framework comprises
two policy layers: (1) a low-level policy that enables the dexterous hand to
manipulate the tool into various configurations for objects of different sizes,
and (2) a high-level policy that defines the tool's goal state and controls the
robotic arm for object-picking tasks. We employ an encoder, trained on
synthetic pointclouds, to estimate the tool's affordance states--specifically,
how different tool configurations (e.g., tweezer opening angles) enable
grasping of objects of varying sizes--from input point clouds, thereby enabling
precise tool manipulation. We also utilize a privilege-informed heuristic
policy to generate replay buffer, improving the training efficiency of the
high-level policy. We validate our approach through real-world experiments,
showing that the robot can effectively manipulate a tweezer-like tool to grasp
objects of diverse shapes and sizes with a 70.8 % success rate. This study
highlights the potential of RL to advance dexterous robotic manipulation of
articulated tools.

### 7. [Friction Estimation for In-Hand Planar Motion](http://arxiv.org/pdf/2507.06824v1)

Authors: Gabriel Arslan Waltersson, Yiannis Karayiannidis

This paper presents a method for online estimation of contact properties
during in-hand sliding manipulation with a parallel gripper. We estimate the
static and Coulomb friction as well as the contact radius from tactile
measurements of contact forces and sliding velocities. The method is validated
in both simulation and real-world experiments. Furthermore, we propose a
heuristic to deal with fast slip-stick dynamics which can adversely affect the
estimation.

### 8. [Toward a Full-Stack Co-Simulation Platform for Testing of Automated Driving Systems](http://arxiv.org/pdf/2507.06884v1)

Authors: Dong Bi, Yongqi Zhao, Zhengguo Gu, Tomislav Mihalj, Jia Hu, Arno Eichberger

Virtual testing has emerged as an effective approach to accelerate the
deployment of automated driving systems. Nevertheless, existing simulation
toolchains encounter difficulties in integrating rapid, automated scenario
generation with simulation environments supporting advanced automated driving
capabilities. To address this limitation, a full-stack toolchain is presented,
enabling automatic scenario generation from real-world datasets and efficient
validation through a co-simulation platform based on CarMaker, ROS, and Apollo.
The simulation results demonstrate the effectiveness of the proposed toolchain.
A demonstration video showcasing the toolchain is available at the provided
link: https://youtu.be/taJw_-CmSiY.

### 9. [ULC: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation](http://arxiv.org/pdf/2507.06905v1)

Authors: Wandong Sun, Luying Feng, Baoshi Cao, Yang Liu, Yaochu Jin, Zongwu Xie

Loco-Manipulation for humanoid robots aims to enable robots to integrate
mobility with upper-body tracking capabilities. Most existing approaches adopt
hierarchical architectures that decompose control into isolated upper-body
(manipulation) and lower-body (locomotion) policies. While this decomposition
reduces training complexity, it inherently limits coordination between
subsystems and contradicts the unified whole-body control exhibited by humans.
We demonstrate that a single unified policy can achieve a combination of
tracking accuracy, large workspace, and robustness for humanoid
loco-manipulation. We propose the Unified Loco-Manipulation Controller (ULC), a
single-policy framework that simultaneously tracks root velocity, root height,
torso rotation, and dual-arm joint positions in an end-to-end manner, proving
the feasibility of unified control without sacrificing performance. We achieve
this unified control through key technologies: sequence skill acquisition for
progressive learning complexity, residual action modeling for fine-grained
control adjustments, command polynomial interpolation for smooth motion
transitions, random delay release for robustness to deploy variations, load
randomization for generalization to external disturbances, and
center-of-gravity tracking for providing explicit policy gradients to maintain
stability. We validate our method on the Unitree G1 humanoid robot with 3-DOF
(degrees-of-freedom) waist. Compared with strong baselines, ULC shows better
tracking performance to disentangled methods and demonstrating larger workspace
coverage. The unified dual-arm tracking enables precise manipulation under
external loads while maintaining coordinated whole-body control for complex
loco-manipulation tasks.

### 10. [Bounomodes: the grazing ox algorithm for exploration of clustered anomalies](http://arxiv.org/pdf/2507.06960v1)

Authors: Samuel Matloob, Ayan Dutta, O. Patrick Kreidl, Swapnonel Roy, Ladislau Bölöni

A common class of algorithms for informative path planning (IPP) follows
boustrophedon ("as the ox turns") patterns, which aim to achieve uniform area
coverage. However, IPP is often applied in scenarios where anomalies, such as
plant diseases, pollution, or hurricane damage, appear in clusters. In such
cases, prioritizing the exploration of anomalous regions over uniform coverage
is beneficial. This work introduces a class of algorithms referred to as
bounom\=odes ("as the ox grazes"), which alternates between uniform
boustrophedon sampling and targeted exploration of detected anomaly clusters.
While uniform sampling can be designed using geometric principles, close
exploration of clusters depends on the spatial distribution of anomalies and
must be learned. In our implementation, the close exploration behavior is
learned using deep reinforcement learning algorithms. Experimental evaluations
demonstrate that the proposed approach outperforms several established
baselines.

### Software Engineering

### 1. [Evaluating Efficiency and Novelty of LLM-Generated Code for Graph Analysis](http://arxiv.org/pdf/2507.06463v1)

Authors: Atieh Barati Nia, Mohammad Dindoost, David A. Bader

Large Language Models (LLMs) are increasingly used to automate software
development, yet most prior evaluations focus on functional correctness or
high-level languages such as Python. We present the first systematic study of
LLMs' ability to generate efficient C implementations of graph-analysis
routines--code that must satisfy the stringent runtime and memory constraints.
Eight state-of-the-art models (OpenAI ChatGPT o3 and o4-mini-high, Anthropic
Claude 4 Sonnet and Sonnet Extended, Google Gemini 2.5 Flash and Pro, xAI Grok
3-Think, and DeepSeek DeepThink R1) are benchmarked by two distinct approaches.
The first approach checks the ability of LLMs in generating an algorithm
outperforming other present algorithms in the benchmark. The second approach
evaluates the ability of LLMs to generate graph algorithms for integration into
the benchmark. Results show that Claude Sonnet 4 Extended achieves the best
result in the case of ready-to-use code generation and efficiency,
outperforming human-written baselines in triangle counting. The study confirms
that contemporary LLMs excel at optimizing and integrating established
algorithms but not inventing novel techniques. We provide prompts, the first
approach's generated code, and measurement scripts to foster reproducible
research.

### 2. [Issue Tracking Ecosystems: Context and Best Practices](http://arxiv.org/pdf/2507.06704v1)

Authors: Lloyd Montgomery

Issue Tracking Systems (ITSs), such as GitHub and Jira, are popular tools
that support Software Engineering (SE) organisations through the management of
``issues'', which represent different SE artefacts such as requirements,
development tasks, and maintenance items. ITSs also support internal linking
between issues, and external linking to other tools and information sources.
This provides SE organisations key forms of documentation, including forwards
and backwards traceability (e.g., Feature Requests linked to sprint releases
and code commits linked to Bug Reports). An Issue Tracking Ecosystem (ITE) is
the aggregate of the central ITS and the related SE artefacts, stakeholders,
and processes -- with an emphasis on how these contextual factors interact with
the ITS. The quality of ITEs is central to the success of these organisations
and their software products. There are challenges, however, within ITEs,
including complex networks of interlinked artefacts and diverse workflows.
While ITSs have been the subject of study in SE research for decades, ITEs as a
whole need further exploration.
  In this thesis, I undertake the challenge of understanding ITEs at a broader
level, addressing these questions regarding complexity and diversity. I
interviewed practitioners and performed archival analysis on a diverse set of
ITSs. These analyses revealed the context-dependent nature of ITE problems,
highlighting the need for context-specific ITE research. While previous work
has produced many solutions to specific ITS problems, these solutions are not
consistently framed in a context-rich and comparable way, leading to a desire
for more aligned solutions across research and practice. To address this
emergent information and lack of alignment, I created the Best Practice
Ontology for ITEs. <... truncated due to arXiv abstract character limit ...>

### 3. [Leveraging LLMs for Semantic Conflict Detection via Unit Test Generation](http://arxiv.org/pdf/2507.06762v1)

Authors: Nathalia Barbosa, Paulo Borba, Léuson Da Silva

Semantic conflicts arise when a developer introduces changes to a codebase
that unintentionally affect the behavior of changes integrated in parallel by
other developers. Traditional merge tools are unable to detect such conflicts,
so complementary tools like SMAT have been proposed. SMAT relies on generating
and executing unit tests: if a test fails on the base version, passes on a
developer's modified version, but fails again after merging with another
developer's changes, a semantic conflict is indicated. While SMAT is effective
at detecting conflicts, it suffers from a high rate of false negatives, partly
due to the limitations of unit test generation tools such as Randoop and
Evosuite. To investigate whether large language models (LLMs) can overcome
these limitations, we propose and integrate a new test generation tool based on
Code Llama 70B into SMAT. We explore the model's ability to generate tests
using different interaction strategies, prompt contents, and parameter
configurations. Our evaluation uses two samples: a benchmark with simpler
systems from related work, and a more significant sample based on complex,
real-world systems. We assess the effectiveness of the new SMAT extension in
detecting conflicts. Results indicate that, although LLM-based test generation
remains challenging and computationally expensive in complex scenarios, there
is promising potential for improving semantic conflict detection.
  --
  Conflitos sem^anticos surgem quando um desenvolvedor introduz mudan\c{c}as em
uma base de c\'odigo que afetam, de forma n~ao intencional, o comportamento de
altera\c{c}~oes integradas em paralelo por outros desenvolvedores. Ferramentas
tradicionais de merge n~ao conseguem detectar esse tipo de conflito, por isso
ferramentas complementares como o SMAT foram propostas. O SMAT depende da
gera\c{c}~ao e execu\c{c}~ao de testes de unidade: se um teste falha na vers~ao
base, passa na vers~ao modificada por um desenvolvedor, mas volta a falhar
ap\'os o merge com as mudan\c{c}as de outro desenvolvedor, um conflito
sem^antico \'e identificado. Embora o SMAT seja eficaz na detec\c{c}~ao de
conflitos, apresenta alta taxa de falsos negativos, em parte devido \`as
limita\c{c}~oes das ferramentas de gera\c{c}~ao de testes como Randoop e
Evosuite. Para investigar se modelos de linguagem de grande porte (LLMs) podem
superar essas limita\c{c}~oes, propomos e integramos ao SMAT uma nova
ferramenta de gera\c{c}~ao de testes baseada no Code Llama 70B. Exploramos a
capacidade do modelo de gerar testes utilizando diferentes estrat\'egias de
intera\c{c}~ao, conte\'udos de prompts e configura\c{c}~oes de par^ametros.
Nossa avalia\c{c}~ao utiliza duas amostras: um benchmark com sistemas mais
simples, usados em trabalhos relacionados, e uma amostra mais significativa
baseada em sistemas complexos e reais. Avaliamos a efic\'acia da nova extens~ao
do SMAT na detec\c{c}~ao de conflitos. Os resultados indicam que, embora a
gera\c{c}~ao de testes por LLM em cen\'arios complexos ainda seja desafiadora e
custosa computacionalmente, h\'a potencial promissor para aprimorar a
detec\c{c}~ao de conflitos sem^anticos.

### 4. [Are They All Good? Evaluating the Quality of CoTs in LLM-based Code Generation](http://arxiv.org/pdf/2507.06980v1)

Authors: Binquan Zhang, Li Zhang, Zhiwen Luo, Yuxin Du, Fang Liu, Song Wang, Lin Shi

Large language models (LLMs) have demonstrated impressive performance in code
generation, particularly when augmented with chain-of-thought (CoT) prompting
techniques. They break down requirements into intermediate reasoning steps,
which act as design rationales to guide LLMs in writing code like human
programmers. Thus, the quality of these steps is crucial for ensuring the
correctness and reliability of the generated code. However, little is known
about the quality of CoT generated by LLMs. To what extent can we trust the
thoughts generated by LLMs? How good are they? This paper empirically explores
the external and internal factors of why LLMs generate unsatisfactory CoTs by
analyzing 1,023 failed code samples on two widely used code generation
benchmarks. We also evaluate their impact on code generation performance by
analyzing 210 CoT-code pairs and refining the unsatisfied CoTs by prompting
LLMs. Our study reveals three key findings: (1) External factors (53.60%), such
as unclear requirements and lack of context, mainly affect CoT quality, while
internal factors (40.10%) stem from LLMs' misunderstanding prompts. (2) Even
when CoTs are correct, 18.5% of the generated code contains errors due to
instruction-following issues; conversely, 11.90% of correct code is paired with
flawed CoTs. (3) Refining low-quality CoTs is feasible, i.e., LLMs improve when
given detailed problem descriptions. These findings highlight key challenges in
CoT-based code generation and suggest directions for improving LLM reasoning
and reliability.

### 5. [Exploring Fairness Interventions in Open Source Projects](http://arxiv.org/pdf/2507.07026v1)

Authors: Sadia Afrin Mim, Fatema Tuz Zohra, Justin Smith, Brittany Johnson

The deployment of biased machine learning (ML) models has resulted in adverse
effects in crucial sectors such as criminal justice and healthcare. To address
these challenges, a diverse range of machine learning fairness interventions
have been developed, aiming to mitigate bias and promote the creation of more
equitable models. Despite the growing availability of these interventions,
their adoption in real-world applications remains limited, with many
practitioners unaware of their existence. To address this gap, we
systematically identified and compiled a dataset of 62 open source fairness
interventions and identified active ones. We conducted an in-depth analysis of
their specifications and features to uncover considerations that may drive
practitioner preference and to identify the software interventions actively
maintained in the open source ecosystem. Our findings indicate that 32% of
these interventions have been actively maintained within the past year, and 50%
of them offer both bias detection and mitigation capabilities, mostly during
inprocessing.

### 6. [TELSAFE: Security Gap Quantitative Risk Assessment Framework](http://arxiv.org/pdf/2507.06497v1)

Authors: Sarah Ali Siddiqui, Chandra Thapa, Derui Wang, Rayne Holland, Wei Shao, Seyit Camtepe, Hajime Suzuki, Rajiv Shah

Gaps between established security standards and their practical
implementation have the potential to introduce vulnerabilities, possibly
exposing them to security risks. To effectively address and mitigate these
security and compliance challenges, security risk management strategies are
essential. However, it must adhere to well-established strategies and industry
standards to ensure consistency, reliability, and compatibility both within and
across organizations. In this paper, we introduce a new hybrid risk assessment
framework called TELSAFE, which employs probabilistic modeling for quantitative
risk assessment and eliminates the influence of expert opinion bias. The
framework encompasses both qualitative and quantitative assessment phases,
facilitating effective risk management strategies tailored to the unique
requirements of organizations. A specific use case utilizing Common
Vulnerabilities and Exposures (CVE)-related data demonstrates the framework's
applicability and implementation in real-world scenarios, such as in the
telecommunications industry.

### 7. [Finding Compiler Bugs through Cross-Language Code Generator and Differential Testing](http://arxiv.org/pdf/2507.06584v1)

Authors: Qiong Feng, Xiaotian Ma, Ziyuan Feng, Marat Akhin, Wei Song, Peng Liang

Compilers play a central role in translating high-level code into executable
programs, making their correctness essential for ensuring code safety and
reliability. While extensive research has focused on verifying the correctness
of compilers for single-language compilation, the correctness of cross-language
compilation - which involves the interaction between two languages and their
respective compilers - remains largely unexplored. To fill this research gap,
we propose CrossLangFuzzer, a novel framework that introduces a universal
intermediate representation (IR) for JVM-based languages and automatically
generates cross-language test programs with diverse type parameters and complex
inheritance structures. After generating the initial IR, CrossLangFuzzer
applies three mutation techniques - LangShuffler, FunctionRemoval, and
TypeChanger - to enhance program diversity. By evaluating both the original and
mutated programs across multiple compiler versions, CrossLangFuzzer
successfully uncovered 10 confirmed bugs in the Kotlin compiler, 4 confirmed
bugs in the Groovy compiler, 7 confirmed bugs in the Scala 3 compiler, 2
confirmed bugs in the Scala 2 compiler, and 1 confirmed bug in the Java
compiler. Among all mutators, TypeChanger is the most effective, detecting 11
of the 24 compiler bugs. Furthermore, we analyze the symptoms and root causes
of cross-compilation bugs, examining the respective responsibilities of
language compilers when incorrect behavior occurs during cross-language
compilation. To the best of our knowledge, this is the firstwork specifically
focused on identifying and diagnosing compiler bugs in cross-language
compilation scenarios. Our research helps to understand these challenges and
contributes to improving compiler correctness in multi-language environments.

### 8. [Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting](http://arxiv.org/pdf/2507.06907v1)

Authors: Linyun Gao, Qiang Wen, Fumio Machida

Autonomous driving is rapidly advancing as a key application of machine
learning, yet ensuring the safety of these systems remains a critical
challenge. Traffic sign recognition, an essential component of autonomous
vehicles, is particularly vulnerable to adversarial attacks that can compromise
driving safety. In this paper, we propose an N-version machine learning (NVML)
framework that integrates a safety-aware weighted soft voting mechanism. Our
approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential
safety risks and assign dynamic, safety-aware weights to the ensemble outputs.
We evaluate the robustness of three-version NVML systems employing various
voting mechanisms against adversarial samples generated using the Fast Gradient
Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental
results demonstrate that our NVML approach significantly enhances the
robustness and safety of traffic sign recognition systems under adversarial
conditions.

### 9. [Enhancing Quantum Software Development Process with Experiment Tracking](http://arxiv.org/pdf/2507.06990v1)

Authors: Mahee Gamage, Otso Kinanen, Jake Muff, Vlad Stirbu

As quantum computing advances from theoretical promise to experimental
reality, the need for rigorous experiment tracking becomes critical. Drawing
inspiration from best practices in machine learning (ML) and artificial
intelligence (AI), we argue that reproducibility, scalability, and
collaboration in quantum research can benefit significantly from structured
tracking workflows. This paper explores the application of MLflow in quantum
research, illustrating how it enables better development practices, experiment
reproducibility, decision making, and cross-domain integration in an
increasingly hybrid classical-quantum landscape.

### 10. [Robust Containerization of the High Angular Resolution Functional Imaging (HARFI) Pipeline](http://arxiv.org/pdf/2507.07010v1)

Authors: Zhiyuan Li, Kurt G. Schilling, Bennett A. Landman

Historically, functional magnetic resonance imaging (fMRI) of the brain has
focused primarily on gray matter, particularly the cortical gray matter and
associated nuclei. However, recent work has demonstrated that functional
activity in white matter also plays a meaningful role in both cognition and
learning. In previous work, we introduced the High Angular Resolution
Functional Imaging (HARFI) pipeline, which demonstrated both local and global
patterns of functional correlation in white matter. Notably, HARFI enabled
exploration of asymmetric voxel-wise correlation using odd-order spherical
harmonics. Although the original implementation of HARFI was released via
GitHub, adoption was limited due to the technical complexity of running the
source code. In this work, we present a robust and efficient containerized
version of the HARFI pipeline, enabling seamless execution across multiple
public datasets. Our goal is to facilitate broader and deeper exploration of
functional white matter architecture, especially through the lens of high
angular resolution functional correlations. The key innovation of this work is
the containerized implementation, which we have made available under a
permissive open-source license to support reproducible and accessible research
practices.

### Social and Information Networks

### 1. [Temporal Motif Participation Profiles for Analyzing Node Similarity in Temporal Networks](http://arxiv.org/pdf/2507.06465v1)

Authors: Maxwell C. Lee, Kevin S. Xu

Temporal networks consisting of timestamped interactions between a set of
nodes provide a useful representation for analyzing complex networked systems
that evolve over time. Beyond pairwise interactions between nodes, temporal
motifs capture patterns of higher-order interactions such as directed triangles
over short time periods. We propose temporal motif participation profiles
(TMPPs) to capture the behavior of nodes in temporal motifs. Two nodes with
similar TMPPs take similar positions within temporal motifs, possibly with
different nodes. TMPPs serve as unsupervised embeddings for nodes in temporal
networks that are directly interpretable, as each entry denotes the frequency
at which a node participates in a particular position in a specific temporal
motif. We demonstrate that clustering TMPPs reveals groups of nodes with
similar roles in a temporal network through simulation experiments and a case
study on a network of militarized interstate disputes.

### 2. [Mitigating Message Imbalance in Fraud Detection with Dual-View Graph Representation Learning](http://arxiv.org/pdf/2507.06469v1)

Authors: Yudan Song, Yuecen Wei, Yuhang Lu, Qingyun Sun, Minglai Shao, Li-e Wang, Chunming Hu, Xianxian Li, Xingcheng Fu

Graph representation learning has become a mainstream method for fraud
detection due to its strong expressive power, which focuses on enhancing node
representations through improved neighborhood knowledge capture. However, the
focus on local interactions leads to imbalanced transmission of global
topological information and increased risk of node-specific information being
overwhelmed during aggregation due to the imbalance between fraud and benign
nodes. In this paper, we first summarize the impact of topology and class
imbalance on downstream tasks in GNN-based fraud detection, as the problem of
imbalanced supervisory messages is caused by fraudsters' topological behavior
obfuscation and identity feature concealment. Based on statistical validation,
we propose a novel dual-view graph representation learning method to mitigate
Message imbalance in Fraud Detection(MimbFD). Specifically, we design a
topological message reachability module for high-quality node representation
learning to penetrate fraudsters' camouflage and alleviate insufficient
propagation. Then, we introduce a local confounding debiasing module to adjust
node representations, enhancing the stable association between node
representations and labels to balance the influence of different classes.
Finally, we conducted experiments on three public fraud datasets, and the
results demonstrate that MimbFD exhibits outstanding performance in fraud
detection.

### 3. [Towards Designing Social Interventions for Online Climate Change Denialism Discussions](http://arxiv.org/pdf/2507.06561v1)

Authors: Ruican zhong, Shruti Phadke, Beth Goldberg, Tanushree Mitra

As conspiracy theories gain traction, it has become crucial to research
effective intervention strategies that can foster evidence and science-based
discussions in conspiracy theory communities online. This study presents a
novel framework using insider language to contest conspiracy theory ideology in
climate change denialism on Reddit. Focusing on discussions in two Reddit
communities, our research investigates reactions to pro-social and
evidence-based intervention messages for two cohorts of users: climate change
deniers and climate change supporters. Specifically, we combine manual and
generative AI-based methods to craft intervention messages and deploy the
interventions as replies on Reddit posts and comments through transparently
labeled bot accounts. On the one hand, we find that evidence-based
interventions with neutral language foster positive engagement, encouraging
open discussions among believers of climate change denialism. On the other,
climate change supporters respond positively, actively participating and
presenting additional evidence. Our study contributes valuable insights into
the process and challenges of automatically delivering interventions in
conspiracy theory communities on social media, and helps inform future research
on social media interventions.

### 4. [5C Prompt Contracts: A Minimalist, Creative-Friendly, Token-Efficient Design Framework for Individual and SME LLM Usage](http://arxiv.org/pdf/2507.07045v1)

Authors: Ugur Ari

The progression from traditional prompt engineering to a more rigorous
discipline of prompt design marks a pivotal shift in human-LLM interaction. As
Large Language Models (LLMs) become increasingly embedded in mission-critical
applications, there emerges a pressing need for frameworks that are not only
explicit and systematic but also minimal enough to remain practical and broadly
accessible. While many existing approaches address prompt structuring through
elaborate Domain-Specific Languages (DSLs) or multi-layered templates, such
methods can impose significant token and cognitive overhead, potentially
constraining the model's creative capacity. In this context, we propose the 5C
Prompt Contract, a framework that distills prompt design into five intuitive
components: Character, Cause, Constraint, Contingency, and Calibration. This
minimal cognitive schema explicitly integrates fallback and output optimization
directives, fostering reliable, interpretable, and creatively flexible AI
interactions. Experimental results demonstrate that the 5C framework
consistently achieves superior input token efficiency while maintaining rich
and consistent outputs across diverse LLM architectures (OpenAI, Anthropic,
DeepSeek, and Gemini), making it particularly suited for individuals and
Small-to-Medium Enterprises (SMEs) with limited AI engineering resources.

### 5. [Graph-based Fake Account Detection: A Survey](http://arxiv.org/pdf/2507.06541v1)

Authors: Ali Safarpoor Dehkordi, Ahad N. Zehmakan

In recent years, there has been a growing effort to develop effective and
efficient algorithms for fake account detection in online social networks. This
survey comprehensively reviews existing methods, with a focus on graph-based
techniques that utilise topological features of social graphs (in addition to
account information, such as their shared contents and profile data) to
distinguish between fake and real accounts. We provide several categorisations
of these methods (for example, based on techniques used, input data, and
detection time), discuss their strengths and limitations, and explain how these
methods connect in the broader context. We also investigate the available
datasets, including both real-world data and synthesised models. We conclude
the paper by proposing several potential avenues for future research.

### 6. [Exploring Public Perceptions of Generative AI in Libraries: A Social Media Analysis of X Discussions](http://arxiv.org/pdf/2507.07047v1)

Authors: Yuan Li, Teja Mandaloju, Haihua Chen

This study investigates public perceptions of generative artificial
intelligence (GenAI) in libraries through a large-scale analysis of posts on X
(formerly Twitter). Using a mixed-method approach that combines temporal trend
analysis, sentiment classification, and social network analysis, this paper
explores how public discourse around GenAI and libraries has evolved over time,
the emotional tones that dominate the conversation, and the key users or
organizations driving engagement. The findings reveal that discussions are
predominantly negative in tone, with surges linked to concerns about ethics and
intellectual property. Furthermore, social network analysis identifies both
institutional authority and individual bridge users who facilitate cross-domain
engagement. The results in this paper contribute to the growing body of
literature on GenAI in the library and GLAM (Galleries, Libraries, Archives,
and Museums) sectors and offer a real-time, public-facing perspective on the
emerging opportunities and concerns GenAI presents.

### 7. [DICE: Data Influence Cascade in Decentralized Learning](http://arxiv.org/pdf/2507.06931v1)

Authors: Tongtian Zhu, Wenhao Li, Can Wang, Fengxiang He

Decentralized learning offers a promising approach to crowdsource data
consumptions and computational workloads across geographically distributed
compute interconnected through peer-to-peer networks, accommodating the
exponentially increasing demands. However, proper incentives are still in
absence, considerably discouraging participation. Our vision is that a fair
incentive mechanism relies on fair attribution of contributions to
participating nodes, which faces non-trivial challenges arising from the
localized connections making influence ``cascade'' in a decentralized network.
To overcome this, we design the first method to estimate \textbf{D}ata
\textbf{I}nfluence \textbf{C}ascad\textbf{E} (DICE) in a decentralized
environment. Theoretically, the framework derives tractable approximations of
influence cascade over arbitrary neighbor hops, suggesting the influence
cascade is determined by an interplay of data, communication topology, and the
curvature of loss landscape. DICE also lays the foundations for applications
including selecting suitable collaborators and identifying malicious behaviors.
Project page is available at https://raiden-zhu.github.io/blog/2025/DICE/.

### Systems and Control

### 1. [Dual State-space Fidelity Blade (D-STAB): A Novel Stealthy Cyber-physical Attack Paradigm](http://arxiv.org/pdf/2507.06492v1)

Authors: Jiajun Shen, Hao Tu, Fengjun Li, Morteza Hashemi, Di Wu, Huazhen Fang

This paper presents a novel cyber-physical attack paradigm, termed the Dual
State-Space Fidelity Blade (D-STAB), which targets the firmware of core
cyber-physical components as a new class of attack surfaces. The D-STAB attack
exploits the information asymmetry caused by the fidelity gap between
high-fidelity and low-fidelity physical models in cyber-physical systems. By
designing precise adversarial constraints based on high-fidelity state-space
information, the attack induces deviations in high-fidelity states that remain
undetected by defenders relying on low-fidelity observations. The effectiveness
of D-STAB is demonstrated through a case study in cyber-physical battery
systems, specifically in an optimal charging task governed by a Battery
Management System (BMS).

### 2. [Effects of Net Metering Policies on Distributed Energy Resource Valuation and Operation](http://arxiv.org/pdf/2507.06595v1)

Authors: Lane D. Smith, Daniel S. Kirschen

Net energy metering has been a successful policy for increasing solar
generation installations and reducing the costs of photovoltaic arrays for
consumers. However, increased maturity of solar technologies and concerns over
cost shifts created by net energy metering have recently caused the policy to
change its incentives. What once favored behind-the-meter solar generation now
is focused on compensating flexible operation. This paper explores the impacts
that different net energy metering policies have on commercial consumers with
various distributed energy resources. We show that the newest iteration of net
energy metering is less beneficial for consumers with only solar generation and
instead favors those that pair energy storage with solar. Though shiftable
flexible demand offers consumers the ability to operate flexibly, the export
prices offered by the latest net energy metering policy provide limited value
to flexible demand.

### 3. [The Small Phase Condition is Necessary for Symmetric Systems](http://arxiv.org/pdf/2507.06617v1)

Authors: Xiaokan Yang, Wei Chen, Li Qiu

In this paper, we show that the small phase condition is both sufficient and
necessary to ensure the feedback stability when the interconnected systems are
symmetric. Such symmetric systems arise in diverse applications. The key lies
in that, for a complex symmetric and semi-sectorial matrix, the transformation
matrix in its generalized sectorial decomposition can be taken to be real. Such
a result fills the gap of phase based necessary condition for the feedback
stability of symmetric systems, and serves as a counterpart of the necessity
result for small gain condition. Moreover, we explore the necessity of small
phase condition for general asymmetric systems. Some insightful results are
presented, which help to clarify the main challenge in the general case.

### 4. [Coordinated Fast Frequency Regulation in Dynamic Virtual Power Plants via Disturbance Estimation](http://arxiv.org/pdf/2507.06713v1)

Authors: Saif Ahmad, Seifeddine Ben Elghali, Hafiz Ahmed

In the context of dynamic virtual power plants (DVPPs), the integration of
frequency containment reserve (FCR) and fast frequency control (FFC) enabled
via local compensation of power imbalance represents a significant advancement
in decentralized frequency regulation. However, they still have to cope with
the limited power and energy capacities associated with commonly available
storage solutions. This work combines a disturbance estimation based
decentralized local control with distributed imbalance compensation in the
event of local shortfall. The layered architecture facilitates fast local
corrections in power setpoints while enabling coordination between neighbouring
DVPP nodes to leverage the aggregated capacity, ensuring scalable and efficient
operation suitable for renewable-heavy future grids. The proposed approach is
validated on an illustrative 4-bus system with a high percentage of renewables.

### 5. [Techno-economic analysis of decarbonized backup power systems using scenario-based stochastic optimization](http://arxiv.org/pdf/2507.06736v1)

Authors: Jonas Schweiger, Ruaridh Macdonald

In the context of growing concerns about power disruptions, grid reliability
and the need for decarbonization, this study evaluates a broad range of clean
backup power systems (BPSs) to replace traditional emergency diesel generators.
A scenario-based stochastic optimization framework using actual load profiles
and outage probabilities is proposed to assess the most promising options from
a pool of 27 technologies. This framework allows a comparison of
cost-effectiveness and environmental impact of individual technologies and
hybrid BPSs across various scenarios. The results highlight the trade-off
between total annual system cost and emissions. Significant emission reductions
can be achieved at moderate cost increases but deep decarbonization levels
incur higher costs. Primary and secondary batteries are included in optimal
clean fuel-based systems across all decarbonization levels, combining
cost-effective power delivery and long-term storage benefits. The findings
highlight the often-overlooked importance of fuel replacement on both emissions
and costs. Among the assessed technologies, ammonia generators and hydrogen
fuel cells combined with secondary iron-air batteries emerge as cost-effective
solutions for achieving decarbonization goals. To ensure a broad range of
applicability, the study outlines the impact of emergency fuel purchases,
varying demand patterns and demand response options on the optimal BPS. The
research findings are valuable for optimizing the design of clean BPSs to
economically meet the needs of many facility types and decarbonization targets.

### 6. [A nonlinear dead-time compensation method for path tracking control](http://arxiv.org/pdf/2507.06935v1)

Authors: Karin Festl, Michael Stolz

In the realm of autonomous vehicle technologies and advanced driver
assistance systems, precise and reliable path tracking controllers are vital
for safe and efficient navigation. However the presence of dead time in the
vehicle control systems poses a challenge to real-world systems. Input and
output delays are caused by factors like sensor processing and mechanical
response and can range up to a few hundred milliseconds. This chapter addresses
the problem of dead time in path tracking control and proposes a method to
compensate the dead time. The proposed solution involves a nonlinear prediction
model, in a structure similar to the Smith predictor, but incorporating the
kinematic behavior of the vehicle plant system. The implementation avoids
numeric integration or optimization, enabling a fast execution. Simulation
tests with various controllers and disturbances, including dead-time
uncertainty, demonstrate the efficacy of the dead-time compensation method.
Results indicate improved control performance in all tested scenarios.

### 7. [Device-to-Device Communication in 5G/6G: Architectural Foundations and Convergence with Enabling Technologies](http://arxiv.org/pdf/2507.06946v1)

Authors: Mohammad Reza Fasihi, Brian L. Mark

Device-to-Device (D2D) communication is a promising solution to meet the
growing demands of 5G and future 6G networks by enabling direct communication
between user devices. It enhances spectral efficiency (SE) and energy
efficiency (EE), reduces latency, and supports proximity-based services. As
wireless systems evolve toward 5G and 6G paradigms, the integration of D2D with
advanced cellular technologies introduces new opportunities and challenges.
This survey paper reviews the architectural foundations of D2D communication
and explores its integration with key 5G/6G enabling technologies. We review
standardization efforts, analyze core challenges, and highlight future research
directions to unlock the full potential of D2D in next-generation wireless
networks.

### 8. [Transferable Parasitic Estimation via Graph Contrastive Learning and Label Rebalancing in AMS Circuits](http://arxiv.org/pdf/2507.06535v1)

Authors: Shan Shen, Shenglu Hua, Jiajun Zou, Jiawei Liu, Jianwang Zhai, Chuan Shi, Wenjian Yu

Graph representation learning on Analog-Mixed Signal (AMS) circuits is
crucial for various downstream tasks, e.g., parasitic estimation. However, the
scarcity of design data, the unbalanced distribution of labels, and the
inherent diversity of circuit implementations pose significant challenges to
learning robust and transferable circuit representations. To address these
limitations, we propose CircuitGCL, a novel graph contrastive learning
framework that integrates representation scattering and label rebalancing to
enhance transferability across heterogeneous circuit graphs. CircuitGCL employs
a self-supervised strategy to learn topology-invariant node embeddings through
hyperspherical representation scattering, eliminating dependency on large-scale
data. Simultaneously, balanced mean squared error (MSE) and softmax
cross-entropy (bsmCE) losses are introduced to mitigate label distribution
disparities between circuits, enabling robust and transferable parasitic
estimation. Evaluated on parasitic capacitance estimation (edge-level task) and
ground capacitance classification (node-level task) across TSMC 28nm AMS
designs, CircuitGCL outperforms all state-of-the-art (SOTA) methods, with the
$R^2$ improvement of $33.64\% \sim 44.20\%$ for edge regression and F1-score
gain of $0.9\times \sim 2.1\times$ for node classification. Our code is
available at
\href{https://anonymous.4open.science/r/CircuitGCL-099B/README.md}{here}.

### 9. [Few-shot Learning on AMS Circuits and Its Application to Parasitic Capacitance Prediction](http://arxiv.org/pdf/2507.06538v1)

Authors: Shan Shen, Yibin Zhang, Hector Rodriguez Rodriguez, Wenjian Yu

Graph representation learning is a powerful method to extract features from
graph-structured data, such as analog/mixed-signal (AMS) circuits. However,
training deep learning models for AMS designs is severely limited by the
scarcity of integrated circuit design data. In this work, we present
CircuitGPS, a few-shot learning method for parasitic effect prediction in AMS
circuits. The circuit netlist is represented as a heterogeneous graph, with the
coupling capacitance modeled as a link. CircuitGPS is pre-trained on link
prediction and fine-tuned on edge regression. The proposed method starts with a
small-hop sampling technique that converts a link or a node into a subgraph.
Then, the subgraph embeddings are learned with a hybrid graph Transformer.
Additionally, CircuitGPS integrates a low-cost positional encoding that
summarizes the positional and structural information of the sampled subgraph.
CircuitGPS improves the accuracy of coupling existence by at least 20\% and
reduces the MAE of capacitance estimation by at least 0.067 compared to
existing methods. Our method demonstrates strong inherent scalability, enabling
direct application to diverse AMS circuit designs through zero-shot learning.
Furthermore, the ablation studies provide valuable insights into graph models
for representation learning.

### 10. [Deep-Learning-Based Pre-Layout Parasitic Capacitance Prediction on SRAM Designs](http://arxiv.org/pdf/2507.06549v1)

Authors: Shan Shen, Dingcheng Yang, Yuyang Xie, Chunyan Pei, Wenjian Yu, Bei Yu

To achieve higher system energy efficiency, SRAM in SoCs is often customized.
The parasitic effects cause notable discrepancies between pre-layout and
post-layout circuit simulations, leading to difficulty in converging design
parameters and excessive design iterations. Is it possible to well predict the
parasitics based on the pre-layout circuit, so as to perform parasitic-aware
pre-layout simulation? In this work, we propose a deep-learning-based 2-stage
model to accurately predict these parasitics in pre-layout stages. The model
combines a Graph Neural Network (GNN) classifier and Multi-Layer Perceptron
(MLP) regressors, effectively managing class imbalance of the net parasitics in
SRAM circuits. We also employ Focal Loss to mitigate the impact of abundant
internal net samples and integrate subcircuit information into the graph to
abstract the hierarchical structure of schematics. Experiments on 4 real SRAM
designs show that our approach not only surpasses the state-of-the-art model in
parasitic prediction by a maximum of 19X reduction of error but also
significantly boosts the simulation process by up to 598X speedup.

### Machine Learning (Statistics Category)

### 1. [Instance-Wise Monotonic Calibration by Constrained Transformation](http://arxiv.org/pdf/2507.06516v1)

Authors: Yunrui Zhang, Gustavo Batista, Salil S. Kanhere

Deep neural networks often produce miscalibrated probability estimates,
leading to overconfident predictions. A common approach for calibration is
fitting a post-hoc calibration map on unseen validation data that transforms
predicted probabilities. A key desirable property of the calibration map is
instance-wise monotonicity (i.e., preserving the ranking of probability
outputs). However, most existing post-hoc calibration methods do not guarantee
monotonicity. Previous monotonic approaches either use an under-parameterized
calibration map with limited expressive ability or rely on black-box neural
networks, which lack interpretability and robustness. In this paper, we propose
a family of novel monotonic post-hoc calibration methods, which employs a
constrained calibration map parameterized linearly with respect to the number
of classes. Our proposed approach ensures expressiveness, robustness, and
interpretability while preserving the relative ordering of the probability
output by formulating the proposed calibration map as a constrained
optimization problem. Our proposed methods achieve state-of-the-art performance
across datasets with different deep neural network models, outperforming
existing calibration methods while being data and computation-efficient. Our
code is available at
https://github.com/YunruiZhang/Calibration-by-Constrained-Transformation

### 2. [Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000](http://arxiv.org/pdf/2507.06619v1)

Authors: Xiaobo Huang, Fang Xie

When applying machine learning to medical image classification, data leakage
is a critical issue. Previous methods, such as adding noise to gradients for
differential privacy, work well on large datasets like MNIST and CIFAR-100, but
fail on small, imbalanced medical datasets like HAM10000. This is because the
imbalanced distribution causes gradients from minority classes to be clipped
and lose crucial information, while majority classes dominate. This leads the
model to fall into suboptimal solutions early. To address this, we propose
SAD-DPSGD, which uses a linear decaying mechanism for noise and clipping
thresholds. By allocating more privacy budget and using higher clipping
thresholds in the initial training phases, the model avoids suboptimal
solutions and enhances performance. Experiments show that SAD-DPSGD outperforms
Auto-DPSGD on HAM10000, improving accuracy by 2.15% under $\epsilon = 3.0$ ,
$\delta = 10^{-3}$.

### 3. [Semi-parametric Functional Classification via Path Signatures Logistic Regression](http://arxiv.org/pdf/2507.06637v1)

Authors: Pengcheng Zeng, Siyuan Jiang

We propose Path Signatures Logistic Regression (PSLR), a semi-parametric
framework for classifying vector-valued functional data with scalar covariates.
Classical functional logistic regression models rely on linear assumptions and
fixed basis expansions, which limit flexibility and degrade performance under
irregular sampling. PSLR overcomes these issues by leveraging truncated path
signatures to construct a finite-dimensional, basis-free representation that
captures nonlinear and cross-channel dependencies. By embedding trajectories as
time-augmented paths, PSLR extracts stable, geometry-aware features that are
robust to sampling irregularity without requiring a common time grid, while
still preserving subject-specific timing patterns. We establish theoretical
guarantees for the existence and consistent estimation of the optimal
truncation order, along with non-asymptotic risk bounds. Experiments on
synthetic and real-world datasets show that PSLR outperforms traditional
functional classifiers in accuracy, robustness, and interpretability,
particularly under non-uniform sampling schemes. Our results highlight the
practical and theoretical benefits of integrating rough path theory into modern
functional data analysis.

### 4. [stCEG: An R Package for Modelling Events over Spatial Areas Using Chain Event Graphs](http://arxiv.org/pdf/2507.06726v1)

Authors: Hollie Calley, Daniel Williamson

stCEG is an R package which allows a user to fully specify a Chain Event
Graph (CEG) model from data and to produce interactive plots. It includes
functions for the user to visualise spatial variables they wish to include in
the model. There is also a web-based graphical user interface (GUI) provided,
increasing ease of use for those without knowledge of R. We demonstrate stCEG
using a dataset of homicides in London, which is included in the package. stCEG
is the first software package for CEGs that allows for full model
customisation.

### 5. [Scalable Gaussian Processes: Advances in Iterative Methods and Pathwise Conditioning](http://arxiv.org/pdf/2507.06839v1)

Authors: Jihao Andreas Lin

Gaussian processes are a powerful framework for uncertainty-aware function
approximation and sequential decision-making. Unfortunately, their classical
formulation does not scale gracefully to large amounts of data and modern
hardware for massively-parallel computation, prompting many researchers to
develop techniques which improve their scalability. This dissertation focuses
on the powerful combination of iterative methods and pathwise conditioning to
develop methodological contributions which facilitate the use of Gaussian
processes in modern large-scale settings. By combining these two techniques
synergistically, expensive computations are expressed as solutions to systems
of linear equations and obtained by leveraging iterative linear system solvers.
This drastically reduces memory requirements, facilitating application to
significantly larger amounts of data, and introduces matrix multiplication as
the main computational operation, which is ideal for modern hardware.

### 6. [Adaptive collaboration for online personalized distributed learning with heterogeneous clients](http://arxiv.org/pdf/2507.06844v1)

Authors: Constantin Philippenko, Batiste Le Bars, Kevin Scaman, Laurent Massoulié

We study the problem of online personalized decentralized learning with $N$
statistically heterogeneous clients collaborating to accelerate local training.
An important challenge in this setting is to select relevant collaborators to
reduce gradient variance while mitigating the introduced bias. To tackle this,
we introduce a gradient-based collaboration criterion, allowing each client to
dynamically select peers with similar gradients during the optimization
process. Our criterion is motivated by a refined and more general theoretical
analysis of the All-for-one algorithm, proved to be optimal in Even et al.
(2022) for an oracle collaboration scheme. We derive excess loss upper-bounds
for smooth objective functions, being either strongly convex, non-convex, or
satisfying the Polyak-Lojasiewicz condition; our analysis reveals that the
algorithm acts as a variance reduction method where the speed-up depends on a
sufficient variance. We put forward two collaboration methods instantiating the
proposed general schema; and we show that one variant preserves the optimality
of All-for-one. We validate our results with experiments on synthetic and real
datasets.

### 7. [Distribution-free inference for LightGBM and GLM with Tweedie loss](http://arxiv.org/pdf/2507.06921v1)

Authors: Alokesh Manna, Aditya Vikram Sett, Dipak K. Dey, Yuwen Gu, Elizabeth D. Schifano, Jichao He

Prediction uncertainty quantification is a key research topic in recent years
scientific and business problems. In insurance industries
(\cite{parodi2023pricing}), assessing the range of possible claim costs for
individual drivers improves premium pricing accuracy. It also enables insurers
to manage risk more effectively by accounting for uncertainty in accident
likelihood and severity. In the presence of covariates, a variety of
regression-type models are often used for modeling insurance claims, ranging
from relatively simple generalized linear models (GLMs) to regularized GLMs to
gradient boosting models (GBMs). Conformal predictive inference has arisen as a
popular distribution-free approach for quantifying predictive uncertainty under
relatively weak assumptions of exchangeability, and has been well studied under
the classic linear regression setting. In this work, we propose new
non-conformity measures for GLMs and GBMs with GLM-type loss. Using regularized
Tweedie GLM regression and LightGBM with Tweedie loss, we demonstrate conformal
prediction performance with these non-conformity measures in insurance claims
data. Our simulation results favor the use of locally weighted Pearson
residuals for LightGBM over other methods considered, as the resulting
intervals maintained the nominal coverage with the smallest average width.

### 8. [Off-Policy Evaluation Under Nonignorable Missing Data](http://arxiv.org/pdf/2507.06961v1)

Authors: Han Wang, Yang Xu, Wenbin Lu, Rui Song

Off-Policy Evaluation (OPE) aims to estimate the value of a target policy
using offline data collected from potentially different policies. In real-world
applications, however, logged data often suffers from missingness. While OPE
has been extensively studied in the literature, a theoretical understanding of
how missing data affects OPE results remains unclear. In this paper, we
investigate OPE in the presence of monotone missingness and theoretically
demonstrate that the value estimates remain unbiased under ignorable
missingness but can be biased under nonignorable (informative) missingness. To
retain the consistency of value estimation, we propose an inverse probability
weighted value estimator and conduct statistical inference to quantify the
uncertainty of the estimates. Through a series of numerical experiments, we
empirically demonstrate that our proposed estimator yields a more reliable
value inference under missing data.

### 9. [AdaDPIGU: Differentially Private SGD with Adaptive Clipping and Importance-Based Gradient Updates for Deep Neural Networks](http://arxiv.org/pdf/2507.06525v1)

Authors: Huiqi Zhang, Fang Xie

Differential privacy has been proven effective for stochastic gradient
descent; however, existing methods often suffer from performance degradation in
high-dimensional settings, as the scale of injected noise increases with
dimensionality. To tackle this challenge, we propose AdaDPIGU--a new
differentially private SGD framework with importance-based gradient updates
tailored for deep neural networks. In the pretraining stage, we apply a
differentially private Gaussian mechanism to estimate the importance of each
parameter while preserving privacy. During the gradient update phase, we prune
low-importance coordinates and introduce a coordinate-wise adaptive clipping
mechanism, enabling sparse and noise-efficient gradient updates. Theoretically,
we prove that AdaDPIGU satisfies $(\varepsilon, \delta)$-differential privacy
and retains convergence guarantees. Extensive experiments on standard
benchmarks validate the effectiveness of AdaDPIGU. All results are reported
under a fixed retention ratio of 60%. On MNIST, our method achieves a test
accuracy of 99.12% under a privacy budget of $\epsilon = 8$, nearly matching
the non-private model. Remarkably, on CIFAR-10, it attains 73.21% accuracy at
$\epsilon = 4$, outperforming the non-private baseline of 71.12%, demonstrating
that adaptive sparsification can enhance both privacy and utility.

### 10. [A Single Merging Suffices: Recovering Server-based Learning Performance in Decentralized Learning](http://arxiv.org/pdf/2507.06542v1)

Authors: Tongtian Zhu, Tianyu Zhang, Mingze Wang, Zhanpeng Zhou, Can Wang

Decentralized learning provides a scalable alternative to traditional
parameter-server-based training, yet its performance is often hindered by
limited peer-to-peer communication. In this paper, we study how communication
should be scheduled over time, including determining when and how frequently
devices synchronize. Our empirical results show that concentrating
communication budgets in the later stages of decentralized training markedly
improves global generalization. Surprisingly, we uncover that fully connected
communication at the final step, implemented by a single global merging, is
sufficient to match the performance of server-based training. We further show
that low communication in decentralized learning preserves the
\textit{mergeability} of local models throughout training. Our theoretical
contributions, which explains these phenomena, are first to establish that the
globally merged model of decentralized SGD can converge faster than centralized
mini-batch SGD. Technically, we novelly reinterpret part of the discrepancy
among local models, which were previously considered as detrimental noise, as
constructive components that accelerate convergence. This work challenges the
common belief that decentralized learning generalizes poorly under data
heterogeneity and limited communication, while offering new insights into model
merging and neural network loss landscapes.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-10 PST.

### 1. [Cognitive bias in clinical large language models](https://www.nature.com/articles/s41746-025-01790-0)

Authors: Arjun Mahajan et al.

### 2. [Deep knowledge tracing and cognitive load estimation for personalized learning path generation using neural network architecture](https://www.nature.com/articles/s41598-025-10497-x)

Authors: Chunyan Tong et al.

### 3. [Real-time monitoring and optimization methods for user-side energy management based on edge computing](https://www.nature.com/articles/s41598-025-07592-4)

Authors: Jisheng Huang et al.

### 4. [Lightweight machine learning framework for efficient DDoS attack detection in IoT networks](https://www.nature.com/articles/s41598-025-10092-0)

Authors: Mamoona Nawaz et al.

### 5. [Aqua-MC as a simple open access code for uncountable runs of AquaCrop](https://www.nature.com/articles/s41598-025-08995-z)

Authors: Vahid Adabi et al.

### 6. [Enabling large language models for real-world materials discovery](https://www.nature.com/articles/s42256-025-01058-y)

Authors: Santiago Miret et al.

### 7. [Energy-efficient data collection optimization for dual-UAV-assisted relay marine networks](https://www.nature.com/articles/s41598-025-10766-9)

Authors: Fukang Deng et al.

### 8. [Project based learning framework integrating industry collaboration to enhance student future readiness in higher education](https://www.nature.com/articles/s41598-025-10385-4)

Authors: Fawad Naseer et al.

### 9. [A Multi-Scale attention network for building extraction from high-resolution remote sensing images](https://www.nature.com/articles/s41598-025-09086-9)

Authors: Jing Chang et al.

### 10. [Automated tick classification using deep learning and its associated challenges in citizen science](https://www.nature.com/articles/s41598-025-10265-x)

Authors: Anna Omazic et al.

### 11. [A deep learning software tool for automated sleep staging in rats via single channel EEG](https://www.nature.com/articles/s44277-025-00035-y)

Authors: Andrew Smith et al.

### 12. [A dual layer secure and energy-efficient model for border surveillance using sea lion inspired strategy in wireless sensor networks](https://www.nature.com/articles/s41598-025-07999-z)

Authors: Jayachandran J et al.

### 13. [A self-learning method with domain knowledge integration for intelligent welding sequence planning](https://www.nature.com/articles/s41598-025-11333-y)

Authors: Weidong Shen et al.

### 14. [SynergyBug: A deep learning approach to autonomous debugging and code remediation](https://www.nature.com/articles/s41598-025-08226-5)

Authors: Hong Chen

### 15. [Leveraging machine learning techniques for image classification and revealing social media insights into human engagement with urban wild spaces](https://www.nature.com/articles/s41598-025-06731-1)

Authors: Haider Khalid et al.

### 16. [Exploiting heart rate variability for driver drowsiness detection using wearable sensors and machine learning](https://www.nature.com/articles/s41598-025-08582-2)

Authors: Zakwan AlArnaout et al.

