# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-06 14:21:06.229623 PST.

### Artificial Intelligence

### 1. SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness

[SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness](http://arxiv.org/pdf/2505.02306v1)

Authors: Junfeng Jiao, Jihyung Park, Yiming Xu, Lucy Atkinson

Despite the abundance of public safety documents and emergency protocols,
most individuals remain ill-equipped to interpret and act on such information
during crises. Traditional emergency decision support systems (EDSS) are
designed for professionals and rely heavily on static documents like PDFs or
SOPs, which are difficult for non-experts to navigate under stress. This gap
between institutional knowledge and public accessibility poses a critical
barrier to effective emergency preparedness and response.
  We introduce SafeMate, a retrieval-augmented AI assistant that delivers
accurate, context-aware guidance to general users in both preparedness and
active emergency scenarios. Built on the Model Context Protocol (MCP), SafeMate
dynamically routes user queries to tools for document retrieval, checklist
generation, and structured summarization. It uses FAISS with cosine similarity
to identify relevant content from trusted sources.

### 2. HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking

[HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](http://arxiv.org/pdf/2505.02322v1)

Authors: Runquan Gui, Zhihai Wang, Jie Wang, Chi Ma, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Defu Lian, Enhong Chen, Feng Wu

Recent advancements have significantly enhanced the performance of large
language models (LLMs) in tackling complex reasoning tasks, achieving notable
success in domains like mathematical and logical reasoning. However, these
methods encounter challenges with complex planning tasks, primarily due to
extended reasoning steps, diverse constraints, and the challenge of handling
multiple distinct sub-tasks. To address these challenges, we propose HyperTree
Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured
planning outlines for effective planning. The hypertree structure enables LLMs
to engage in hierarchical thinking by flexibly employing the divide-and-conquer
strategy, effectively breaking down intricate reasoning steps, accommodating
diverse constraints, and managing multiple distinct sub-tasks in a
well-organized manner. We further introduce an autonomous planning framework
that completes the planning process by iteratively refining and expanding the
hypertree-structured planning outlines. Experiments demonstrate the
effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner
benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement
over o1-preview.

### 3. Task-Oriented Semantic Communication in Large Multimodal Models-based Vehicle Networks

[Task-Oriented Semantic Communication in Large Multimodal Models-based Vehicle Networks](http://arxiv.org/pdf/2505.02413v1)

Authors: Baoxia Du, Hongyang Du, Dusit Niyato, Ruidong Li

Task-oriented semantic communication has emerged as a fundamental approach
for enhancing performance in various communication scenarios. While recent
advances in Generative Artificial Intelligence (GenAI), such as Large Language
Models (LLMs), have been applied to semantic communication designs, the
potential of Large Multimodal Models (LMMs) remains largely unexplored. In this
paper, we investigate an LMM-based vehicle AI assistant using a Large Language
and Vision Assistant (LLaVA) and propose a task-oriented semantic communication
framework to facilitate efficient interaction between users and cloud servers.
To reduce computational demands and shorten response time, we optimize LLaVA's
image slicing to selectively focus on areas of utmost interest to users.
Additionally, we assess the importance of image patches by combining objective
and subjective user attention, adjusting energy usage for transmitting semantic
information. This strategy optimizes resource utilization, ensuring precise
transmission of critical information. We construct a Visual Question Answering
(VQA) dataset for traffic scenarios to evaluate effectiveness. Experimental
results show that our semantic communication framework significantly increases
accuracy in answering questions under the same channel conditions, performing
particularly well in environments with poor Signal-to-Noise Ratios (SNR).
Accuracy can be improved by 13.4% at an SNR of 12dB and 33.1% at 10dB,
respectively.

### 4. MSFNet-CPD: Multi-Scale Cross-Modal Fusion Network for Crop Pest Detection

[MSFNet-CPD: Multi-Scale Cross-Modal Fusion Network for Crop Pest Detection](http://arxiv.org/pdf/2505.02441v1)

Authors: Jiaqi Zhang, Zhuodong Liu, Kejian Yu

Accurate identification of agricultural pests is essential for crop
protection but remains challenging due to the large intra-class variance and
fine-grained differences among pest species. While deep learning has advanced
pest detection, most existing approaches rely solely on low-level visual
features and lack effective multi-modal integration, leading to limited
accuracy and poor interpretability. Moreover, the scarcity of high-quality
multi-modal agricultural datasets further restricts progress in this field. To
address these issues, we construct two novel multi-modal benchmarks-CTIP102 and
STIP102-based on the widely-used IP102 dataset, and introduce a Multi-scale
Cross-Modal Fusion Network (MSFNet-CPD) for robust pest detection. Our approach
enhances visual quality via a super-resolution reconstruction module, and feeds
both the original and reconstructed images into the network to improve clarity
and detection performance. To better exploit semantic cues, we propose an
Image-Text Fusion (ITF) module for joint modeling of visual and textual
features, and an Image-Text Converter (ITC) that reconstructs fine-grained
details across multiple scales to handle challenging backgrounds. Furthermore,
we introduce an Arbitrary Combination Image Enhancement (ACIE) strategy to
generate a more complex and diverse pest detection dataset, MTIP102, improving
the model's generalization to real-world scenarios. Extensive experiments
demonstrate that MSFNet-CPD consistently outperforms state-of-the-art methods
on multiple pest detection benchmarks. All code and datasets will be made
publicly available at: https://github.com/Healer-ML/MSFNet-CPD.

### 5. Agentic Neurodivergence as a Contingent Solution to the AI Alignment Problem

[Agentic Neurodivergence as a Contingent Solution to the AI Alignment Problem](http://arxiv.org/pdf/2505.02581v1)

Authors: Alberto Hernández-Espinosa, Felipe S. Abrahão, Olaf Witkowski, Hector Zenil

The AI alignment problem, which focusses on ensuring that artificial
intelligence (AI), including AGI and ASI, systems act according to human
values, presents profound challenges. With the progression from narrow AI to
Artificial General Intelligence (AGI) and Superintelligence, fears about
control and existential risk have escalated. This paper demonstrates that
achieving complete alignment is inherently unattainable due to mathematical
principles rooted in the foundations of predicate logic and computability, in
particular Turing's computational universality, G\"odel's incompleteness and
Chaitin's randomness. Instead, we argue that embracing AI misalignment or
agent's `neurodivergence' as a contingent strategy, defined as fostering a
dynamic ecosystem of competing, partially aligned agents, is a possible only
viable path to mitigate risks. Through mathematical proofs and an experimental
design, we explore how misalignment may serve and should be promoted as a
counterbalancing mechanism to team up with whichever agents are most aligned AI
to human values, ensuring that no single system dominates destructively. The
main premise of our contribution is that misalignment is inevitable because
full AI-human alignment is a mathematical impossibility from Turing-complete
systems which we also prove in this paper, a feature then inherited to AGI and
ASI systems. We introduce and test `change-of-opinion' attacks based on this
kind of perturbation and intervention analysis to study how agents may
neutralise friendly or unfriendly AIs through cooperation, competition or
malice.

### 6. A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law

[A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law](http://arxiv.org/pdf/2505.02665v1)

Authors: Qianjun Pan, Wenkai Ji, Yuyang Ding, Junsong Li, Shilian Chen, Junyi Wang, Jie Zhou, Qin Chen, Min Zhang, Yulan Wu, Liang He

This survey explores recent advancements in reasoning large language models
(LLMs) designed to mimic "slow thinking" - a reasoning process inspired by
human cognition, as described in Kahneman's Thinking, Fast and Slow. These
models, like OpenAI's o1, focus on scaling computational resources dynamically
during complex tasks, such as math reasoning, visual reasoning, medical
diagnosis, and multi-agent debates. We present the development of reasoning
LLMs and list their key technologies. By synthesizing over 100 studies, it
charts a path toward LLMs that combine human-like deep thinking with scalable
efficiency for reasoning. The review breaks down methods into three categories:
(1) test-time scaling dynamically adjusts computation based on task complexity
via search and sampling, dynamic verification; (2) reinforced learning refines
decision-making through iterative improvement leveraging policy networks,
reward models, and self-evolution strategies; and (3) slow-thinking frameworks
(e.g., long CoT, hierarchical processes) that structure problem-solving with
manageable steps. The survey highlights the challenges and further directions
of this domain. Understanding and advancing the reasoning abilities of LLMs is
crucial for unlocking their full potential in real-world applications, from
scientific discovery to decision support systems.

### 7. Local Markov Equivalence and Local Causal Discovery for Identifying Controlled Direct Effects

[Local Markov Equivalence and Local Causal Discovery for Identifying Controlled Direct Effects](http://arxiv.org/pdf/2505.02781v1)

Authors: Timothée Loranchet, Charles K. Assaad

Understanding and identifying controlled direct effects (CDEs) is crucial
across numerous scientific domains, including public health. While existing
methods can identify these effects from causal directed acyclic graphs (DAGs),
the true underlying structure is often unknown in practice. Essential graphs,
which represent a Markov equivalence class of DAGs characterized by the same
set of d-separations, provide a more practical and realistic alternative.
However, learning the full essential graph is computationally intensive and
typically depends on strong, untestable assumptions. In this work, we
characterize a local class of graphs, defined relative to a target variable,
that share a specific subset of d-separations, and introduce a graphical
representation of this class, called the local essential graph (LEG). We then
present LocPC, a novel algorithm designed to recover the LEG from an observed
distribution using only local conditional independence tests. Building on
LocPC, we propose LocPC-CDE, an algorithm that discovers the portion of the LEG
that is sufficient to identify a CDE, bypassing the need of retrieving the full
essential graph. Compared to global methods, our algorithms require less
conditional independence tests and operate under weaker assumptions while
maintaining theoretical guarantees.

### 8. LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery

[LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery](http://arxiv.org/pdf/2505.02829v1)

Authors: Jerome Quenum, Wen-Han Hsieh, Tsung-Han Wu, Ritwik Gupta, Trevor Darrell, David M. Chan

Segmentation models can recognize a pre-defined set of objects in images.
However, models that can reason over complex user queries that implicitly refer
to multiple objects of interest are still in their infancy. Recent advances in
reasoning segmentation--generating segmentation masks from complex, implicit
query text--demonstrate that vision-language models can operate across an open
domain and produce reasonable outputs. However, our experiments show that such
models struggle with complex remote-sensing imagery. In this work, we introduce
LISAt, a vision-language model designed to describe complex remote-sensing
scenes, answer questions about them, and segment objects of interest. We
trained LISAt on a new curated geospatial reasoning-segmentation dataset, GRES,
with 27,615 annotations over 9,205 images, and a multimodal pretraining
dataset, PreGRES, containing over 1 million question-answer pairs. LISAt
outperforms existing geospatial foundation models such as RS-GPT4V by over
10.04 % (BLEU-4) on remote-sensing description tasks, and surpasses
state-of-the-art open-domain models on reasoning segmentation tasks by 143.36 %
(gIoU). Our model, datasets, and code are available at
https://lisat-bair.github.io/LISAt/

### 9. Adaptive Scoring and Thresholding with Human Feedback for Robust Out-of-Distribution Detection

[Adaptive Scoring and Thresholding with Human Feedback for Robust Out-of-Distribution Detection](http://arxiv.org/pdf/2505.02299v1)

Authors: Daisuke Yamada, Harit Vishwakarma, Ramya Korlakai Vinayak

Machine Learning (ML) models are trained on in-distribution (ID) data but
often encounter out-of-distribution (OOD) inputs during deployment -- posing
serious risks in safety-critical domains. Recent works have focused on
designing scoring functions to quantify OOD uncertainty, with score thresholds
typically set based solely on ID data to achieve a target true positive rate
(TPR), since OOD data is limited before deployment. However, these TPR-based
thresholds leave false positive rates (FPR) uncontrolled, often resulting in
high FPRs where OOD points are misclassified as ID. Moreover, fixed scoring
functions and thresholds lack the adaptivity needed to handle newly observed,
evolving OOD inputs, leading to sub-optimal performance. To address these
challenges, we propose a human-in-the-loop framework that \emph{safely updates
both scoring functions and thresholds on the fly} based on real-world OOD
inputs. Our method maximizes TPR while strictly controlling FPR at all times,
even as the system adapts over time. We provide theoretical guarantees for FPR
control under stationary conditions and present extensive empirical evaluations
on OpenOOD benchmarks to demonstrate that our approach outperforms existing
methods by achieving higher TPRs while maintaining FPR control.

### 10. What Is AI Safety? What Do We Want It to Be?

[What Is AI Safety? What Do We Want It to Be?](http://arxiv.org/pdf/2505.02313v1)

Authors: Jacqueline Harding, Cameron Domenico Kirk-Giannini

The field of AI safety seeks to prevent or reduce the harms caused by AI
systems. A simple and appealing account of what is distinctive of AI safety as
a field holds that this feature is constitutive: a research project falls
within the purview of AI safety just in case it aims to prevent or reduce the
harms caused by AI systems. Call this appealingly simple account The Safety
Conception of AI safety. Despite its simplicity and appeal, we argue that The
Safety Conception is in tension with at least two trends in the ways AI safety
researchers and organizations think and talk about AI safety: first, a tendency
to characterize the goal of AI safety research in terms of catastrophic risks
from future systems; second, the increasingly popular idea that AI safety can
be thought of as a branch of safety engineering. Adopting the methodology of
conceptual engineering, we argue that these trends are unfortunate: when we
consider what concept of AI safety it would be best to have, there are
compelling reasons to think that The Safety Conception is the answer.
Descriptively, The Safety Conception allows us to see how work on topics that
have historically been treated as central to the field of AI safety is
continuous with work on topics that have historically been treated as more
marginal, like bias, misinformation, and privacy. Normatively, taking The
Safety Conception seriously means approaching all efforts to prevent or
mitigate harms from AI systems based on their merits rather than drawing
arbitrary distinctions between them.

### Hardware Architecture

### 1. NeuroSim V1.5: Improved Software Backbone for Benchmarking Compute-in-Memory Accelerators with Device and Circuit-level Non-idealities

[NeuroSim V1.5: Improved Software Backbone for Benchmarking Compute-in-Memory Accelerators with Device and Circuit-level Non-idealities](http://arxiv.org/pdf/2505.02314v1)

Authors: James Read, Ming-Yen Lee, Wei-Hsing Huang, Yuan-Chun Luo, Anni Lu, Shimeng Yu

The exponential growth of artificial intelligence (AI) applications has
exposed the inefficiency of conventional von Neumann architectures, where
frequent data transfers between compute units and memory create significant
energy and latency bottlenecks. Analog Computing-in-Memory (ACIM) addresses
this challenge by performing multiply-accumulate (MAC) operations directly in
the memory arrays, substantially reducing data movement. However, designing
robust ACIM accelerators requires accurate modeling of device- and
circuit-level non-idealities. In this work, we present NeuroSim V1.5,
introducing several key advances: (1) seamless integration with TensorRT's
post-training quantization flow enabling support for more neural networks
including transformers, (2) a flexible noise injection methodology built on
pre-characterized statistical models, making it straightforward to incorporate
data from SPICE simulations or silicon measurements, (3) expanded device
support including emerging non-volatile capacitive memories, and (4) up to 6.5x
faster runtime than NeuroSim V1.4 through optimized behavioral simulation. The
combination of these capabilities uniquely enables systematic design space
exploration across both accuracy and hardware efficiency metrics. Through
multiple case studies, we demonstrate optimization of critical design
parameters while maintaining network accuracy. By bridging high-fidelity noise
modeling with efficient simulation, NeuroSim V1.5 advances the design and
validation of next-generation ACIM accelerators. All NeuroSim versions are
available open-source at https://github.com/neurosim/NeuroSim.

### 2. Open Challenges for a Production-ready Cloud Environment on top of RISC-V hardware

[Open Challenges for a Production-ready Cloud Environment on top of RISC-V hardware](http://arxiv.org/pdf/2505.02650v1)

Authors: Aaron Call, Ramon Nou, Guillem Senabre

As part of the Vitamin-V European project, we have built a prototype of a
RISC-V cluster managed by OpenStack, with the goal of realizing a functional
RISC-V cloud ecosystem. In this poster we explain the hardware and software
challenges encountered while porting some elements of OpenStack. We also
discuss the current performance gaps that challenge a performance-ready cloud
environment over such new ISA, an essential element to fulfill in order to
achieve european technological sovereignty.

### 3. Machine-Learning-Powered Neural Interfaces for Smart Prosthetics and Diagnostics

[Machine-Learning-Powered Neural Interfaces for Smart Prosthetics and Diagnostics](http://arxiv.org/pdf/2505.02516v1)

Authors: MohammadAli Shaeri, Jinhan Liu, Mahsa Shoaran

Advanced neural interfaces are transforming applications ranging from
neuroscience research to diagnostic tools (for mental state recognition, tremor
and seizure detection) as well as prosthetic devices (for motor and
communication recovery). By integrating complex functions into miniaturized
neural devices, these systems unlock significant opportunities for personalized
assistive technologies and adaptive therapeutic interventions. Leveraging
high-density neural recordings, on-site signal processing, and machine learning
(ML), these interfaces extract critical features, identify disease
neuro-markers, and enable accurate, low-latency neural decoding. This
integration facilitates real-time interpretation of neural signals, adaptive
modulation of brain activity, and efficient control of assistive devices.
Moreover, the synergy between neural interfaces and ML has paved the way for
self-sufficient, ubiquitous platforms capable of operating in diverse
environments with minimal hardware costs and external dependencies. In this
work, we review recent advancements in AI-driven decoding algorithms and
energy-efficient System-on-Chip (SoC) platforms for next-generation
miniaturized neural devices. These innovations highlight the potential for
developing intelligent neural interfaces, addressing critical challenges in
scalability, reliability, interpretability, and user adaptability.

### Computational Complexity

### 1. PLS-completeness of string permutations

[PLS-completeness of string permutations](http://arxiv.org/pdf/2505.02622v1)

Authors: Dominik Scheder, Johannes Tantow

Bitstrings can be permuted via permutations and compared via the
lexicographic order. In this paper we study the complexity of finding a minimum
of a bitstring via given permutations. As a global optima is known to be
NP-complete, we study the local optima via the class PLS and show hardness for
PLS. Additionally, we show that even for one permutation the global
optimization is NP-complete and give a formula that has these permutation as
symmetries. This answers an open question inspired from Kolodziejczyk and
Thapen and stated at the SAT and interactions seminar in Dagstuhl.

### 2. On the Equivalence of Gaussian Graphical Models Defined on Complete Bipartite Graphs

[On the Equivalence of Gaussian Graphical Models Defined on Complete Bipartite Graphs](http://arxiv.org/pdf/2505.02384v1)

Authors: Mehdi Molkaraie

This paper introduces two Gaussian graphical models defined on complete
bipartite graphs. We show that the determinants of the precision matrices
associated with the models are equal up to scale, where the scale factor only
depends on model parameters. In this context, we will introduce a notion of
``equivalence" between the two Gaussian graphical models. This equivalence has
two key applications: first, it can significantly reduce the complexity of
computing the exact value of the determinant, and second, it enables the
derivation of closed-form expressions for the determinants in certain special
cases.

### Computational Engineering

### 1. Predicting the Dynamics of Complex System via Multiscale Diffusion Autoencoder

[Predicting the Dynamics of Complex System via Multiscale Diffusion Autoencoder](http://arxiv.org/pdf/2505.02450v1)

Authors: Ruikun Li, Jingwen Cheng, Huandong Wang, Qingmin Liao, Yong Li

Predicting the dynamics of complex systems is crucial for various scientific
and engineering applications. The accuracy of predictions depends on the
model's ability to capture the intrinsic dynamics. While existing methods
capture key dynamics by encoding a low-dimensional latent space, they overlook
the inherent multiscale structure of complex systems, making it difficult to
accurately predict complex spatiotemporal evolution. Therefore, we propose a
Multiscale Diffusion Prediction Network (MDPNet) that leverages the multiscale
structure of complex systems to discover the latent space of intrinsic
dynamics. First, we encode multiscale features through a multiscale diffusion
autoencoder to guide the diffusion model for reliable reconstruction. Then, we
introduce an attention-based graph neural ordinary differential equation to
model the co-evolution across different scales. Extensive evaluations on
representative systems demonstrate that the proposed method achieves an average
prediction error reduction of 53.23% compared to baselines, while also
exhibiting superior robustness and generalization.

### 2. Data Compression for Time Series Modelling: A Case Study of Smart Grid Demand Forecasting

[Data Compression for Time Series Modelling: A Case Study of Smart Grid Demand Forecasting](http://arxiv.org/pdf/2505.02606v1)

Authors: Mikkel Bue Lykkegaard, Svend Vendelbo Nielsen, Akanksha Upadhyay, Mikkel Bendixen Copeland, Philipp Trénell

Efficient time series forecasting is essential for smart energy systems,
enabling accurate predictions of energy demand, renewable resource
availability, and grid stability. However, the growing volume of high-frequency
data from sensors and IoT devices poses challenges for storage and
transmission. This study explores Discrete Wavelet Transform (DWT)-based data
compression as a solution to these challenges while ensuring forecasting
accuracy. A case study of a seawater supply system in Hirtshals, Denmark,
operating under dynamic weather, operational schedules, and seasonal trends, is
used for evaluation.
  Biorthogonal wavelets of varying orders were applied to compress data at
different rates. Three forecasting models - Ordinary Least Squares (OLS),
XGBoost, and the Time Series Dense Encoder (TiDE) - were tested to assess the
impact of compression on forecasting performance. Lossy compression rates up to
$r_{\mathrm{lossy}} = 0.999$ were analyzed, with the Normalized Mutual
Information (NMI) metric quantifying the relationship between compression and
information retention. Results indicate that wavelet-based compression can
retain essential features for accurate forecasting when applied carefully.
  XGBoost proved highly robust to compression artifacts, maintaining stable
performance across diverse compression rates. In contrast, OLS demonstrated
sensitivity to smooth wavelets and high compression rates, while TiDE showed
some variability but remained competitive. This study highlights the potential
of wavelet-based compression for scalable, efficient data management in smart
energy systems without sacrificing forecasting accuracy. The findings are
relevant to other fields requiring high-frequency time series forecasting,
including climate modeling, water supply systems, and industrial operations.

### Computational Geometry

### 1. Guarding Terrains with Guards on a Line

[Guarding Terrains with Guards on a Line](http://arxiv.org/pdf/2505.02373v1)

Authors: Byeonguk Kang, Hwi Kim, Hee-Kap Ahn

Given an $x$-monotone polygonal chain $T$ with $n$ vertices, and an integer
$k$, we consider the problem of finding the lowest horizontal line $L$ lying
above $T$ with $k$ point guards lying on $L$, so that every point on the chain
is \emph{visible} from some guard. A natural optimization is to minimize the
$y$-coordinate of $L$. We present an algorithm for finding the optimal
placements of $L$ and $k$ point guards for $T$ in $O(k^2\lambda_{k-1}(n)\log
n)$ time for even numbers $k\ge 2$, and in $O(k^2\lambda_{k-2}(n)\log n)$ time
for odd numbers $k \ge 3$, where $\lambda_{s}(n)$ is the length of the longest
$(n,s)$-Davenport-Schinzel sequence. We also study a variant with an additional
requirement that $T$ is partitioned into $k$ subchains, each subchain is paired
with exactly one guard, and every point on a subchain is visible from its
paired guard. When $L$ is fixed, we can place the minimum number of guards in
$O(n)$ time. When the number $k$ of guards is fixed, we can find an optimal
placement of $L$ with $k$ point guards lying on $L$ in $O(kn)$ time.

### 2. Trajectory Minimum Touching Ball

[Trajectory Minimum Touching Ball](http://arxiv.org/pdf/2505.02472v1)

Authors: Jeff M. Phillips, Jens Kristian Refsgaard Schou

We present algorithms to find the minimum radius sphere that intersects every
trajectory in a set of $n$ trajectories composed of at most $k$ line segments
each. When $k=1$, we can reduce the problem to the LP-type framework to achieve
a linear time complexity. For $k \geq 4$ we provide a trajectory configuration
with unbounded LP-type complexity, but also present an almost
$O\left((nk)^2\log n\right)$ algorithm through the farthest line segment
Voronoi diagrams. If we tolerate a relative approximation, we can reduce to
time near-linear in $n$.

### Computation and Language

### 1. Invoke Interfaces Only When Needed: Adaptive Invocation for Large Language Models in Question Answering

[Invoke Interfaces Only When Needed: Adaptive Invocation for Large Language Models in Question Answering](http://arxiv.org/pdf/2505.02311v1)

Authors: Jihao Zhao, Chunlai Zhou, Biao Qin

The collaborative paradigm of large and small language models (LMs)
effectively balances performance and cost, yet its pivotal challenge lies in
precisely pinpointing the moment of invocation when hallucinations arise in
small LMs. Previous optimization efforts primarily focused on post-processing
techniques, which were separate from the reasoning process of LMs, resulting in
high computational costs and limited effectiveness. In this paper, we propose a
practical invocation evaluation metric called AttenHScore, which calculates the
accumulation and propagation of hallucinations during the generation process of
small LMs, continuously amplifying potential reasoning errors. By dynamically
adjusting the detection threshold, we achieve more accurate real-time
invocation of large LMs. Additionally, considering the limited reasoning
capacity of small LMs, we leverage uncertainty-aware knowledge reorganization
to assist them better capture critical information from different text chunks.
Extensive experiments reveal that our AttenHScore outperforms most baseline in
enhancing real-time hallucination detection capabilities across multiple QA
datasets, especially when addressing complex queries. Moreover, our strategies
eliminate the need for additional model training and display flexibility in
adapting to various transformer-based LMs.

### 2. SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning

[SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](http://arxiv.org/pdf/2505.02363v1)

Authors: Tianjian Li, Daniel Khashabi

Aligning language models with human preferences relies on pairwise preference
datasets. While some studies suggest that on-policy data consistently
outperforms off -policy data for preference learning, others indicate that the
advantages of on-policy data may be task-dependent, highlighting the need for a
systematic exploration of their interplay.
  In this work, we show that on-policy and off-policy data offer complementary
strengths in preference optimization: on-policy data is particularly effective
for reasoning tasks like math and coding, while off-policy data performs better
on open-ended tasks such as creative writing and making personal
recommendations. Guided by these findings, we introduce SIMPLEMIX, an approach
to combine the complementary strengths of on-policy and off-policy preference
learning by simply mixing these two data sources. Our empirical results across
diverse tasks and benchmarks demonstrate that SIMPLEMIX substantially improves
language model alignment. Specifically, SIMPLEMIX improves upon on-policy DPO
and off-policy DPO by an average of 6.03% on Alpaca Eval 2.0. Moreover, it
outperforms prior approaches that are much more complex in combining on- and
off-policy data, such as HyPO and DPO-Mix-P, by an average of 3.05%.

### 3. Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs

[Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs](http://arxiv.org/pdf/2505.02456v1)

Authors: Elisa Forcada Rodríguez, Olatz Perez-de-Viñaspre, Jon Ander Campos, Dietrich Klakow, Vagrant Gautam

One of the goals of fairness research in NLP is to measure and mitigate
stereotypical biases that are propagated by NLP systems. However, such work
tends to focus on single axes of bias (most often gender) and the English
language. Addressing these limitations, we contribute the first study of
multilingual intersecting country and gender biases, with a focus on occupation
recommendations generated by large language models. We construct a benchmark of
prompts in English, Spanish and German, where we systematically vary country
and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite
of 5 Llama-based models on this benchmark, finding that LLMs encode significant
gender and country biases. Notably, we find that even when models show parity
for gender or country individually, intersectional occupational biases based on
both country and gender persist. We also show that the prompting language
significantly affects bias, and instruction-tuned models consistently
demonstrate the lowest and most stable levels of bias. Our findings highlight
the need for fairness researchers to use intersectional and multilingual lenses
in their work.

### 4. Data Augmentation With Back translation for Low Resource languages: A case of English and Luganda

[Data Augmentation With Back translation for Low Resource languages: A case of English and Luganda](http://arxiv.org/pdf/2505.02463v1)

Authors: Richard Kimera, Dongnyeong Heo, Daniela N. Rim, Heeyoul Choi

In this paper,we explore the application of Back translation (BT) as a
semi-supervised technique to enhance Neural Machine Translation(NMT) models for
the English-Luganda language pair, specifically addressing the challenges faced
by low-resource languages. The purpose of our study is to demonstrate how BT
can mitigate the scarcity of bilingual data by generating synthetic data from
monolingual corpora. Our methodology involves developing custom NMT models
using both publicly available and web-crawled data, and applying Iterative and
Incremental Back translation techniques. We strategically select datasets for
incremental back translation across multiple small datasets, which is a novel
element of our approach. The results of our study show significant
improvements, with translation performance for the English-Luganda pair
exceeding previous benchmarks by more than 10 BLEU score units across all
translation directions. Additionally, our evaluation incorporates comprehensive
assessment metrics such as SacreBLEU, ChrF2, and TER, providing a nuanced
understanding of translation quality. The conclusion drawn from our research
confirms the efficacy of BT when strategically curated datasets are utilized,
establishing new performance benchmarks and demonstrating the potential of BT
in enhancing NMT models for low-resource languages.

### 5. Proper Name Diacritization for Arabic Wikipedia: A Benchmark Dataset

[Proper Name Diacritization for Arabic Wikipedia: A Benchmark Dataset](http://arxiv.org/pdf/2505.02656v1)

Authors: Rawan Bondok, Mayar Nassar, Salam Khalifa, Kurt Micallaf, Nizar Habash

Proper names in Arabic Wikipedia are frequently undiacritized, creating
ambiguity in pronunciation and interpretation, especially for transliterated
named entities of foreign origin. While transliteration and diacritization have
been well-studied separately in Arabic NLP,their intersection remains
underexplored. In this paper, we introduce a new manually diacritized dataset
of Arabic proper names of various origins with their English Wikipedia
equivalent glosses, and present the challenges and guidelines we followed to
create it. We benchmark GPT-4o on the task of recovering full diacritization
given the undiacritized Arabic and English forms, and analyze its performance.
Achieving 73% accuracy, our results underscore both the difficulty of the task
and the need for improved models and resources. We release our dataset to
facilitate further research on Arabic Wikipedia proper name diacritization.

### 6. A Survey on Progress in LLM Alignment from the Perspective of Reward Design

[A Survey on Progress in LLM Alignment from the Perspective of Reward Design](http://arxiv.org/pdf/2505.02666v1)

Authors: Miaomiao Ji, Yanqiu Wu, Zhibin Wu, Shoujin Wang, Jian Yang, Mark Dras, Usman Naseem

The alignment of large language models (LLMs) with human values and
intentions represents a core challenge in current AI research, where reward
mechanism design has become a critical factor in shaping model behavior. This
study conducts a comprehensive investigation of reward mechanisms in LLM
alignment through a systematic theoretical framework, categorizing their
development into three key phases: (1) feedback (diagnosis), (2) reward design
(prescription), and (3) optimization (treatment). Through a four-dimensional
analysis encompassing construction basis, format, expression, and granularity,
this research establishes a systematic classification framework that reveals
evolutionary trends in reward modeling. The field of LLM alignment faces
several persistent challenges, while recent advances in reward design are
driving significant paradigm shifts. Notable developments include the
transition from reinforcement learning-based frameworks to novel optimization
paradigms, as well as enhanced capabilities to address complex alignment
scenarios involving multimodal integration and concurrent task coordination.
Finally, this survey outlines promising future research directions for LLM
alignment through innovative reward design strategies.

### 7. Sailing AI by the Stars: A Survey of Learning from Rewards in Post-Training and Test-Time Scaling of Large Language Models

[Sailing AI by the Stars: A Survey of Learning from Rewards in Post-Training and Test-Time Scaling of Large Language Models](http://arxiv.org/pdf/2505.02686v1)

Authors: Xiaobao Wu

Recent developments in Large Language Models (LLMs) have shifted from
pre-training scaling to post-training and test-time scaling. Across these
developments, a key unified paradigm has arisen: Learning from Rewards, where
reward signals act as the guiding stars to steer LLM behavior. It has
underpinned a wide range of prevalent techniques, such as reinforcement
learning (in RLHF, DPO, and GRPO), reward-guided decoding, and post-hoc
correction. Crucially, this paradigm enables the transition from passive
learning from static data to active learning from dynamic feedback. This endows
LLMs with aligned preferences and deep reasoning capabilities. In this survey,
we present a comprehensive overview of the paradigm of learning from rewards.
We categorize and analyze the strategies under this paradigm across training,
inference, and post-inference stages. We further discuss the benchmarks for
reward models and the primary applications. Finally we highlight the challenges
and future directions. We maintain a paper collection at
https://github.com/bobxwu/learning-from-rewards-llm-papers.

### 8. ReplaceMe: Network Simplification via Layer Pruning and Linear Transformations

[ReplaceMe: Network Simplification via Layer Pruning and Linear Transformations](http://arxiv.org/pdf/2505.02819v1)

Authors: Dmitriy Shopkhoev, Ammar Ali, Magauiya Zhussip, Valentin Malykh, Stamatios Lefkimmiatis, Nikos Komodakis, Sergey Zagoruyko

We introduce ReplaceMe, a generalized training-free depth pruning method that
effectively replaces transformer blocks with a linear operation, while
maintaining high performance for low compression ratios. In contrast to
conventional pruning approaches that require additional training or
fine-tuning, our approach requires only a small calibration dataset that is
used to estimate a linear transformation to approximate the pruned blocks. This
estimated linear mapping can be seamlessly merged with the remaining
transformer blocks, eliminating the need for any additional network parameters.
Our experiments show that ReplaceMe consistently outperforms other
training-free approaches and remains highly competitive with state-of-the-art
pruning methods that involve extensive retraining/fine-tuning and architectural
modifications. Applied to several large language models (LLMs), ReplaceMe
achieves up to 25% pruning while retaining approximately 90% of the original
model's performance on open benchmarks - without any training or healing steps,
resulting in minimal computational overhead (see Fig.1). We provide an
open-source library implementing ReplaceMe alongside several state-of-the-art
depth pruning techniques, available at this repository.

### 9. Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition

[Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition](http://arxiv.org/pdf/2505.02304v1)

Authors: Siyu Liang, Yunan Li, Wentian Xin, Huizhou Chen, Xujie Liu, Kang Liu, Qiguang Miao

Sign language recognition (SLR) faces fundamental challenges in creating
accurate annotations due to the inherent complexity of simultaneous manual and
non-manual signals. To the best of our knowledge, this is the first work to
integrate generative large language models (LLMs) into SLR tasks. We propose a
novel Generative Sign-description Prompts Multi-positive Contrastive learning
(GSP-MC) method that leverages retrieval-augmented generation (RAG) with
domain-specific LLMs, incorporating multi-step prompt engineering and
expert-validated sign language corpora to produce precise multipart
descriptions. The GSP-MC method also employs a dual-encoder architecture to
bidirectionally align hierarchical skeleton features with multiple text
descriptions (global, synonym, and part level) through probabilistic matching.
Our approach combines global and part-level losses, optimizing KL divergence to
ensure robust alignment across all relevant text-skeleton pairs while capturing
both sign-level semantics and detailed part dynamics. Experiments demonstrate
state-of-the-art performance against existing methods on the Chinese SLR500
(reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's
cross-lingual effectiveness highlight its potential for developing inclusive
communication technologies.

### 10. Bielik 11B v2 Technical Report

[Bielik 11B v2 Technical Report](http://arxiv.org/pdf/2505.02410v1)

Authors: Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoździej, Remigiusz Kinas

We present Bielik 11B v2, a state-of-the-art language model optimized for
Polish text processing. Built on the Mistral 7B v0.2 architecture and scaled to
11B parameters using depth up-scaling, this model demonstrates exceptional
performance across Polish language benchmarks while maintaining strong
cross-lingual capabilities. We introduce two key technical innovations:
Weighted Instruction Cross-Entropy Loss, which optimizes learning across
diverse instruction types by assigning quality-based weights to training
examples, and Adaptive Learning Rate, which dynamically adjusts based on
context length. Comprehensive evaluation across multiple benchmarks
demonstrates that Bielik 11B v2 outperforms many larger models, including those
with 2-6 times more parameters, and significantly surpasses other specialized
Polish language models on tasks ranging from linguistic understanding to
complex reasoning. The model's parameter efficiency and extensive quantization
options enable deployment across various hardware configurations, advancing
Polish language AI capabilities and establishing new benchmarks for
resource-efficient language modeling in less-represented languages.

### Cryptography and Security

### 1. An End-to-End Model For Logits Based Large Language Models Watermarking

[An End-to-End Model For Logits Based Large Language Models Watermarking](http://arxiv.org/pdf/2505.02344v1)

Authors: Kahim Wong, Jicheng Zhou, Jiantao Zhou, Yain-Whar Si

The rise of LLMs has increased concerns over source tracing and copyright
protection for AIGC, highlighting the need for advanced detection technologies.
Passive detection methods usually face high false positives, while active
watermarking techniques using logits or sampling manipulation offer more
effective protection. Existing LLM watermarking methods, though effective on
unaltered content, suffer significant performance drops when the text is
modified and could introduce biases that degrade LLM performance in downstream
tasks. These methods fail to achieve an optimal tradeoff between text quality
and robustness, particularly due to the lack of end-to-end optimization of the
encoder and decoder. In this paper, we introduce a novel end-to-end logits
perturbation method for watermarking LLM-generated text. By jointly
optimization, our approach achieves a better balance between quality and
robustness. To address non-differentiable operations in the end-to-end training
pipeline, we introduce an online prompting technique that leverages the
on-the-fly LLM as a differentiable surrogate. Our method achieves superior
robustness, outperforming distortion-free methods by 37-39% under paraphrasing
and 17.2% on average, while maintaining text quality on par with these
distortion-free methods in terms of text perplexity and downstream tasks. Our
method can be easily generalized to different LLMs.

### 2. Moneros Decentralized P2P Exchanges: Functionality, Adoption, and Privacy Risks

[Moneros Decentralized P2P Exchanges: Functionality, Adoption, and Privacy Risks](http://arxiv.org/pdf/2505.02392v1)

Authors: Yannik Kopyciok, Friedhelm Victor, Stefan Schmid

Privacy-focused cryptocurrencies like Monero remain popular, despite
increasing regulatory scrutiny that has led to their delisting from major
centralized exchanges. The latter also explains the recent popularity of
decentralized exchanges (DEXs) with no centralized ownership structures. These
platforms typically leverage peer-to-peer (P2P) networks, promising secure and
anonymous asset trading. However, questions of liability remain, and the
academic literature lacks comprehensive insights into the functionality,
trading activity, and privacy claims of these P2P platforms. In this paper, we
provide an early systematization of the current landscape of decentralized
peer-to-peer exchanges within the Monero ecosystem. We examine several recently
developed DEX platforms, analyzing their popularity, functionality,
architectural choices, and potential weaknesses. We further identify and report
on a privacy vulnerability in the recently popularized Haveno exchange,
demonstrating that certain Haveno trades could be detected, allowing
transactions to be linked across the Monero and Bitcoin blockchains. We hope
that our findings can nourish the discussion in the research community about
more secure designs, and provide insights for regulators.

### 3. Encrypted Federated Search Using Homomorphic Encryption

[Encrypted Federated Search Using Homomorphic Encryption](http://arxiv.org/pdf/2505.02409v1)

Authors: Om Rathod, Aastha Baid, Aswani Kumar Cherukuri

The sharing of information between agencies is effective in dealing with
cross-jurisdictional criminal activities; however, such sharing is often
restricted due to concerns about data privacy, ownership, and compliance.
Towards this end, this work has introduced a privacy-preserving federated
search system that allows law enforcement agencies to conduct queries on
encrypted criminal databases by utilizing Homomorphic Encryption (HE). The key
innovation here is the ability to execute encrypted queries across distributed
databases, without the decryption of the data, thus preserving end-to-end
confidentiality. In essence, this approach meets stringent privacy requirements
in the interests of national security and regulatory compliance. The system
incorporates the CKKS and BFV scheme embedded within TenSEAL, with each agency
holding its key pair in a centralized key management table. In this federated
search, encrypted queries are computed on the server side, and only authorized
clients can decrypt the computed results. The matching of agencies is flexible
for working in real-time while at the same time being secure and scalable while
preserving control over data and the integrity of the process. Experimental
results demonstrate the model. This paper also provide the implementation code
and other details.

### 4. Targeted Fuzzing for Unsafe Rust Code: Leveraging Selective Instrumentation

[Targeted Fuzzing for Unsafe Rust Code: Leveraging Selective Instrumentation](http://arxiv.org/pdf/2505.02464v1)

Authors: David Paaßen, Jens-Rene Giesen, Lucas Davi

Rust is a promising programming language that focuses on concurrency,
usability, and security. It is used in production code by major industry
players and got recommended by government bodies. Rust provides strong security
guarantees achieved by design utilizing the concepts of ownership and
borrowing. However, Rust allows programmers to write unsafe code which is not
subject to the strict Rust security policy. Empirical studies show that
security issues in practice always involve code written in unsafe Rust.
  In this paper, we present the first approach that utilizes selective code
coverage feedback to focus the fuzzing efforts on unsafe Rust code. Our
approach significantly improves the efficiency when fuzzing Rust programs and
does not require additional computational resources while fuzz testing the
target. To quantify the impact of partial code instrumentation, we implement
our approach by extending the capabilities of the Rust compiler toolchain. We
present an automated approach to detect unsafe and safe code components to
decide which parts of the program a fuzzer should focus on when running a
fuzzing campaign to find vulnerabilities in Rust programs. Our approach is
fully compatible with existing fuzzing implementations and does not require
complex manual work, thus retaining the existing high usability standard.
Focusing on unsafe code, our implementation allows us to generate inputs that
trigger more unsafe code locations with statistical significance and therefore
is able to detect potential vulnerabilities in a shorter time span while
imposing no performance overhead during fuzzing itself.

### 5. Attestable builds: compiling verifiable binaries on untrusted systems using trusted execution environments

[Attestable builds: compiling verifiable binaries on untrusted systems using trusted execution environments](http://arxiv.org/pdf/2505.02521v1)

Authors: Daniel Hugenroth, Mario Lins, René Mayrhofer, Alastair Beresford

In this paper we present attestable builds, a new paradigm to provide strong
source-to-binary correspondence in software artifacts. We tackle the challenge
of opaque build pipelines that disconnect the trust between source code, which
can be understood and audited, and the final binary artifact, which is
difficult to inspect. Our system uses modern trusted execution environments
(TEEs) and sandboxed build containers to provide strong guarantees that a given
artifact was correctly built from a specific source code snapshot. As such it
complements existing approaches like reproducible builds which typically
require time-intensive modifications to existing build configurations and
dependencies, and require independent parties to continuously build and verify
artifacts. In comparison, an attestable build requires only minimal changes to
an existing project, and offers nearly instantaneous verification of the
correspondence between a given binary and the source code and build pipeline
used to construct it. We evaluate it by building open-source software libraries
- focusing on projects which are important to the trust chain and those which
have proven difficult to be built deterministically. Overall, the overhead (42
seconds start-up latency and 14% increase in build duration) is small in
comparison to the overall build time. Importantly, our prototype builds even
complex projects such as LLVM Clang without requiring any modifications to
their source code and build scripts. Finally, we formally model and verify the
attestable build design to demonstrate its security against well-resourced
adversaries.

### 6. SoK: Stealing Cars Since Remote Keyless Entry Introduction and How to Defend From It

[SoK: Stealing Cars Since Remote Keyless Entry Introduction and How to Defend From It](http://arxiv.org/pdf/2505.02713v1)

Authors: Tommaso Bianchi, Alessandro Brighente, Mauro Conti, Edoardo Pavan

Remote Keyless Entry (RKE) systems have been the target of thieves since
their introduction in automotive industry. Robberies targeting vehicles and
their remote entry systems are booming again without a significant advancement
from the industrial sector being able to protect against them. Researchers and
attackers continuously play cat and mouse to implement new methodologies to
exploit weaknesses and defense strategies for RKEs. In this fragment, different
attacks and defenses have been discussed in research and industry without
proper bridging. In this paper, we provide a Systematization Of Knowledge (SOK)
on RKE and Passive Keyless Entry and Start (PKES), focusing on their history
and current situation, ranging from legacy systems to modern web-based ones. We
provide insight into vehicle manufacturers' technologies and attacks and
defense mechanisms involving them. To the best of our knowledge, this is the
first comprehensive SOK on RKE systems, and we address specific research
questions to understand the evolution and security status of such systems. By
identifying the weaknesses RKE still faces, we provide future directions for
security researchers and companies to find viable solutions to address old
attacks, such as Relay and RollJam, as well as new ones, like API
vulnerabilities.

### 7. Acoustic Side-Channel Attacks on a Computer Mouse

[Acoustic Side-Channel Attacks on a Computer Mouse](http://arxiv.org/pdf/2505.02725v1)

Authors: Mauro Conti, Marin Duroyon, Gabriele Orazi, Gene Tsudik

Acoustic Side-Channel Attacks (ASCAs) extract sensitive information by using
audio emitted from a computing devices and their peripherals. Attacks targeting
keyboards are popular and have been explored in the literature. However,
similar attacks targeting other human interface peripherals, such as computer
mice, are under-explored. To this end, this paper considers security leakage
via acoustic signals emanating from normal mouse usage. We first confirm
feasibility of such attacks by showing a proof-of-concept attack that
classifies four mouse movements with 97% accuracy in a controlled environment.
We then evolve the attack towards discerning twelve unique mouse movements
using a smartphone to record the experiment. Using Machine Learning (ML)
techniques, the model is trained on an experiment with six participants to be
generalizable and discern among twelve movements with 94% accuracy. In
addition, we experiment with an attack that detects a user action of closing a
full-screen window on a laptop. Achieving an accuracy of 91%, this experiment
highlights exploiting audio leakage from computer mouse movements in a
realistic scenario.

### 8. A Slicing-Based Approach for Detecting and Patching Vulnerable Code Clones

[A Slicing-Based Approach for Detecting and Patching Vulnerable Code Clones](http://arxiv.org/pdf/2505.02349v1)

Authors: Hakam Alomari, Christopher Vendome, Hilal Gyawali

Code cloning is a common practice in software development, but it poses
significant security risks by propagating vulnerabilities across cloned
segments. To address this challenge, we introduce srcVul, a scalable, precise
detection approach that combines program slicing with Locality-Sensitive
Hashing to identify vulnerable code clones and recommend patches. srcVul builds
a database of vulnerability-related slices by analyzing known vulnerable
programs and their corresponding patches, indexing each slice's unique
structural characteristics as a vulnerability slicing vector. During clone
detection, srcVul efficiently matches slicing vectors from target programs with
those in the database, recommending patches upon identifying similarities. Our
evaluation of srcVul against three state-of-the-art vulnerable clone detectors
demonstrates its accuracy, efficiency, and scalability, achieving 91% precision
and 75% recall on established vulnerability databases and open-source
repositories. These results highlight srcVul's effectiveness in detecting
complex vulnerability patterns across diverse codebases.

### 9. Dynamic Graph-based Fingerprinting of In-browser Cryptomining

[Dynamic Graph-based Fingerprinting of In-browser Cryptomining](http://arxiv.org/pdf/2505.02493v1)

Authors: Tanapoom Sermchaiwong, Jiasi Shen

The decentralized and unregulated nature of cryptocurrencies, combined with
their monetary value, has made them a vehicle for various illicit activities.
One such activity is cryptojacking, an attack that uses stolen computing
resources to mine cryptocurrencies without consent for profit. In-browser
cryptojacking malware exploits high-performance web technologies like
WebAssembly to mine cryptocurrencies directly within the browser without file
downloads. Although existing methods for cryptomining detection report high
accuracy and low overhead, they are often susceptible to various forms of
obfuscation, and due to the limited variety of cryptomining scripts in the
wild, standard code obfuscation methods present a natural and appealing
solution to avoid detection. To address these limitations, we propose using
instruction-level data-flow graphs to detect cryptomining behavior. Data-flow
graphs offer detailed structural insights into a program's computations, making
them suitable for characterizing proof-of-work algorithms, but they can be
difficult to analyze due to their large size and susceptibility to noise and
fragmentation under obfuscation. We present two techniques to simplify and
compare data-flow graphs: (1) a graph simplification algorithm to reduce the
computational burden of processing large and granular data-flow graphs while
preserving local substructures; and (2) a subgraph similarity measure, the
n-fragment inclusion score, based on fragment inclusion that is robust against
noise and obfuscation. Using data-flow graphs as computation fingerprints, our
detection framework PoT (Proof-of-Theft) was able to achieve high detection
accuracy against standard obfuscations, outperforming existing detection
methods. Moreover, PoT uses generic data-flow properties that can be applied to
other platforms more susceptible to cryptojacking such as servers and data
centers.

### 10. Advancing Email Spam Detection: Leveraging Zero-Shot Learning and Large Language Models

[Advancing Email Spam Detection: Leveraging Zero-Shot Learning and Large Language Models](http://arxiv.org/pdf/2505.02362v1)

Authors: Ghazaleh SHirvani, Saeid Ghasemshirazi

Email spam detection is a critical task in modern communication systems,
essential for maintaining productivity, security, and user experience.
Traditional machine learning and deep learning approaches, while effective in
static settings, face significant limitations in adapting to evolving spam
tactics, addressing class imbalance, and managing data scarcity. These
challenges necessitate innovative approaches that reduce dependency on
extensive labeled datasets and frequent retraining. This study investigates the
effectiveness of Zero-Shot Learning using FLAN-T5, combined with advanced
Natural Language Processing (NLP) techniques such as BERT for email spam
detection. By employing BERT to preprocess and extract critical information
from email content, and FLAN-T5 to classify emails in a Zero-Shot framework,
the proposed approach aims to address the limitations of traditional spam
detection systems. The integration of FLAN-T5 and BERT enables robust spam
detection without relying on extensive labeled datasets or frequent retraining,
making it highly adaptable to unseen spam patterns and adversarial
environments. This research highlights the potential of leveraging zero-shot
learning and NLPs for scalable and efficient spam detection, providing insights
into their capability to address the dynamic and challenging nature of spam
detection tasks.

### Computer Vision and Pattern Recognition

### 1. TeDA: Boosting Vision-Lanuage Models for Zero-Shot 3D Object Retrieval via Testing-time Distribution Alignment

[TeDA: Boosting Vision-Lanuage Models for Zero-Shot 3D Object Retrieval via Testing-time Distribution Alignment](http://arxiv.org/pdf/2505.02325v1)

Authors: Zhichuan Wang, Yang Zhou, Jinhai Xiang, Yulong Wang, Xinwei He

Learning discriminative 3D representations that generalize well to unknown
testing categories is an emerging requirement for many real-world 3D
applications. Existing well-established methods often struggle to attain this
goal due to insufficient 3D training data from broader concepts. Meanwhile,
pre-trained large vision-language models (e.g., CLIP) have shown remarkable
zero-shot generalization capabilities. Yet, they are limited in extracting
suitable 3D representations due to substantial gaps between their 2D training
and 3D testing distributions. To address these challenges, we propose
Testing-time Distribution Alignment (TeDA), a novel framework that adapts a
pretrained 2D vision-language model CLIP for unknown 3D object retrieval at
test time. To our knowledge, it is the first work that studies the test-time
adaptation of a vision-language model for 3D feature learning. TeDA projects 3D
objects into multi-view images, extracts features using CLIP, and refines 3D
query embeddings with an iterative optimization strategy by confident
query-target sample pairs in a self-boosting manner. Additionally, TeDA
integrates textual descriptions generated by a multimodal language model
(InternVL) to enhance 3D object understanding, leveraging CLIP's aligned
feature space to fuse visual and textual cues. Extensive experiments on four
open-set 3D object retrieval benchmarks demonstrate that TeDA greatly
outperforms state-of-the-art methods, even those requiring extensive training.
We also experimented with depth maps on Objaverse-LVIS, further validating its
effectiveness. Code is available at https://github.com/wangzhichuan123/TeDA.

### 2. 6D Pose Estimation on Spoons and Hands

[6D Pose Estimation on Spoons and Hands](http://arxiv.org/pdf/2505.02335v1)

Authors: Kevin Tan, Fan Yang, Yuhao Chen

Accurate dietary monitoring is essential for promoting healthier eating
habits. A key area of research is how people interact and consume food using
utensils and hands. By tracking their position and orientation, it is possible
to estimate the volume of food being consumed, or monitor eating behaviours,
highly useful insights into nutritional intake that can be more reliable than
popular methods such as self-reporting. Hence, this paper implements a system
that analyzes stationary video feed of people eating, using 6D pose estimation
to track hand and spoon movements to capture spatial position and orientation.
In doing so, we examine the performance of two state-of-the-art (SOTA) video
object segmentation (VOS) models, both quantitatively and qualitatively, and
identify main sources of error within the system.

### 3. Quaternion Infrared Visible Image Fusion

[Quaternion Infrared Visible Image Fusion](http://arxiv.org/pdf/2505.02364v1)

Authors: Weihua Yang, Yicong Zhou

Visible images provide rich details and color information only under
well-lighted conditions while infrared images effectively highlight thermal
targets under challenging conditions such as low visibility and adverse
weather. Infrared-visible image fusion aims to integrate complementary
information from infrared and visible images to generate a high-quality fused
image. Existing methods exhibit critical limitations such as neglecting color
structure information in visible images and performance degradation when
processing low-quality color-visible inputs. To address these issues, we
propose a quaternion infrared-visible image fusion (QIVIF) framework to
generate high-quality fused images completely in the quaternion domain. QIVIF
proposes a quaternion low-visibility feature learning model to adaptively
extract salient thermal targets and fine-grained texture details from input
infrared and visible images respectively under diverse degraded conditions.
QIVIF then develops a quaternion adaptive unsharp masking method to adaptively
improve high-frequency feature enhancement with balanced illumination. QIVIF
further proposes a quaternion hierarchical Bayesian fusion model to integrate
infrared saliency and enhanced visible details to obtain high-quality fused
images. Extensive experiments across diverse datasets demonstrate that our
QIVIF surpasses state-of-the-art methods under challenging low-visibility
conditions.

### 4. Quaternion Multi-focus Color Image Fusion

[Quaternion Multi-focus Color Image Fusion](http://arxiv.org/pdf/2505.02365v1)

Authors: Weihua Yang, Yicong Zhou

Multi-focus color image fusion refers to integrating multiple partially
focused color images to create a single all-in-focus color image. However,
existing methods struggle with complex real-world scenarios due to limitations
in handling color information and intricate textures. To address these
challenges, this paper proposes a quaternion multi-focus color image fusion
framework to perform high-quality color image fusion completely in the
quaternion domain. This framework introduces 1) a quaternion sparse
decomposition model to jointly learn fine-scale image details and structure
information of color images in an iterative fashion for high-precision focus
detection, 2) a quaternion base-detail fusion strategy to individually fuse
base-scale and detail-scale results across multiple color images for preserving
structure and detail information, and 3) a quaternion structural similarity
refinement strategy to adaptively select optimal patches from initial fusion
results and obtain the final fused result for preserving fine details and
ensuring spatially consistent outputs. Extensive experiments demonstrate that
the proposed framework outperforms state-of-the-art methods.

### 5. Uncertainty-Weighted Image-Event Multimodal Fusion for Video Anomaly Detection

[Uncertainty-Weighted Image-Event Multimodal Fusion for Video Anomaly Detection](http://arxiv.org/pdf/2505.02393v1)

Authors: Sungheon Jeong, Jihong Park, Mohsen Imani

Most existing video anomaly detectors rely solely on RGB frames, which lack
the temporal resolution needed to capture abrupt or transient motion cues, key
indicators of anomalous events. To address this limitation, we propose
Image-Event Fusion for Video Anomaly Detection (IEF-VAD), a framework that
synthesizes event representations directly from RGB videos and fuses them with
image features through a principled, uncertainty-aware process. The system (i)
models heavy-tailed sensor noise with a Student`s-t likelihood, deriving
value-level inverse-variance weights via a Laplace approximation; (ii) applies
Kalman-style frame-wise updates to balance modalities over time; and (iii)
iteratively refines the fused latent state to erase residual cross-modal noise.
Without any dedicated event sensor or frame-level labels, IEF-VAD sets a new
state of the art across multiple real-world anomaly detection benchmarks. These
findings highlight the utility of synthetic event representations in
emphasizing motion cues that are often underrepresented in RGB frames, enabling
accurate and robust video understanding across diverse applications without
requiring dedicated event sensors. Code and models are available at
https://github.com/EavnJeong/IEF-VAD.

### 6. Token Coordinated Prompt Attention is Needed for Visual Prompting

[Token Coordinated Prompt Attention is Needed for Visual Prompting](http://arxiv.org/pdf/2505.02406v1)

Authors: Zichen Liu, Xu Zou, Gang Hua, Jiahuan Zhou

Visual prompting techniques are widely used to efficiently fine-tune
pretrained Vision Transformers (ViT) by learning a small set of shared prompts
for all tokens. However, existing methods overlook the unique roles of
different tokens in conveying discriminative information and interact with all
tokens using the same prompts, thereby limiting the representational capacity
of ViT. This often leads to indistinguishable and biased prompt-extracted
features, hindering performance. To address this issue, we propose a
plug-and-play Token Coordinated Prompt Attention (TCPA) module, which assigns
specific coordinated prompts to different tokens for attention-based
interactions. Firstly, recognizing the distinct functions of CLS and image
tokens-global information aggregation and local feature extraction, we
disentangle the prompts into CLS Prompts and Image Prompts, which interact
exclusively with CLS tokens and image tokens through attention mechanisms. This
enhances their respective discriminative abilities. Furthermore, as different
image tokens correspond to distinct image patches and contain diverse
information, we employ a matching function to automatically assign coordinated
prompts to individual tokens. This enables more precise attention interactions,
improving the diversity and representational capacity of the extracted
features. Extensive experiments across various benchmarks demonstrate that TCPA
significantly enhances the diversity and discriminative power of the extracted
features. The code is available at
https://github.com/zhoujiahuan1991/ICML2025-TCPA.

### 7. Recent Advances in Out-of-Distribution Detection with CLIP-Like Models: A Survey

[Recent Advances in Out-of-Distribution Detection with CLIP-Like Models: A Survey](http://arxiv.org/pdf/2505.02448v1)

Authors: Chaohua Li, Enhao Zhang, Chuanxing Geng, Songcan Chen

Out-of-distribution detection (OOD) is a pivotal task for real-world
applications that trains models to identify samples that are distributionally
different from the in-distribution (ID) data during testing. Recent advances in
AI, particularly Vision-Language Models (VLMs) like CLIP, have revolutionized
OOD detection by shifting from traditional unimodal image detectors to
multimodal image-text detectors. This shift has inspired extensive research;
however, existing categorization schemes (e.g., few- or zero-shot types) still
rely solely on the availability of ID images, adhering to a unimodal paradigm.
To better align with CLIP's cross-modal nature, we propose a new categorization
framework rooted in both image and text modalities. Specifically, we categorize
existing methods based on how visual and textual information of OOD data is
utilized within image + text modalities, and further divide them into four
groups: OOD Images (i.e., outliers) Seen or Unseen, and OOD Texts (i.e.,
learnable vectors or class names) Known or Unknown, across two training
strategies (i.e., train-free or training-required). More importantly, we
discuss open problems in CLIP-like OOD detection and highlight promising
directions for future research, including cross-domain integration, practical
applications, and theoretical understanding.

### 8. Ming-Lite-Uni: Advancements in Unified Architecture for Natural Multimodal Interaction

[Ming-Lite-Uni: Advancements in Unified Architecture for Natural Multimodal Interaction](http://arxiv.org/pdf/2505.02471v1)

Authors: Biao Gong, Cheng Zou, Dandan Zheng, Hu Yu, Jingdong Chen, Jianxin Sun, Junbo Zhao, Jun Zhou, Kaixiang Ji, Lixiang Ru, Libin Wang, Qingpei Guo, Rui Liu, Weilong Chai, Xinyu Xiao, Ziyuan Huang

We introduce Ming-Lite-Uni, an open-source multimodal framework featuring a
newly designed unified visual generator and a native multimodal autoregressive
model tailored for unifying vision and language. Specifically, this project
provides an open-source implementation of the integrated MetaQueries and
M2-omni framework, while introducing the novel multi-scale learnable tokens and
multi-scale representation alignment strategy. By leveraging a fixed MLLM and a
learnable diffusion model, Ming-Lite-Uni enables native multimodal AR models to
perform both text-to-image generation and instruction based image editing
tasks, expanding their capabilities beyond pure visual understanding. Our
experimental results demonstrate the strong performance of Ming-Lite-Uni and
illustrate the impressive fluid nature of its interactive process. All code and
model weights are open-sourced to foster further exploration within the
community. Notably, this work aligns with concurrent multimodal AI milestones -
such as ChatGPT-4o with native image generation updated in March 25, 2025 -
underscoring the broader significance of unified models like Ming-Lite-Uni on
the path toward AGI. Ming-Lite-Uni is in alpha stage and will soon be further
refined.

### 9. Finger Pose Estimation for Under-screen Fingerprint Sensor

[Finger Pose Estimation for Under-screen Fingerprint Sensor](http://arxiv.org/pdf/2505.02481v1)

Authors: Xiongjun Guan, Zhiyu Pan, Jianjiang Feng, Jie Zhou

Two-dimensional pose estimation plays a crucial role in fingerprint
recognition by facilitating global alignment and reduce pose-induced
variations. However, existing methods are still unsatisfactory when handling
with large angle or small area inputs. These limitations are particularly
pronounced on fingerprints captured by under-screen fingerprint sensors in
smartphones. In this paper, we present a novel dual-modal input based network
for under-screen fingerprint pose estimation. Our approach effectively
integrates two distinct yet complementary modalities: texture details extracted
from ridge patches through the under-screen fingerprint sensor, and rough
contours derived from capacitive images obtained via the touch screen. This
collaborative integration endows our network with more comprehensive and
discriminative information, substantially improving the accuracy and stability
of pose estimation. A decoupled probability distribution prediction task is
designed, instead of the traditional supervised forms of numerical regression
or heatmap voting, to facilitate the training process. Additionally, we
incorporate a Mixture of Experts (MoE) based feature fusion mechanism and a
relationship driven cross-domain knowledge transfer strategy to further
strengthen feature extraction and fusion capabilities. Extensive experiments
are conducted on several public datasets and two private datasets. The results
indicate that our method is significantly superior to previous state-of-the-art
(SOTA) methods and remarkably boosts the recognition ability of fingerprint
recognition algorithms. Our code is available at
https://github.com/XiongjunGuan/DRACO.

### 10. Text to Image Generation and Editing: A Survey

[Text to Image Generation and Editing: A Survey](http://arxiv.org/pdf/2505.02527v1)

Authors: Pengfei Yang, Ngai-Man Cheung, Xinda Ma

Text-to-image generation (T2I) refers to the text-guided generation of
high-quality images. In the past few years, T2I has attracted widespread
attention and numerous works have emerged. In this survey, we comprehensively
review 141 works conducted from 2021 to 2024. First, we introduce four
foundation model architectures of T2I (autoregression, non-autoregression, GAN
and diffusion) and the commonly used key technologies (autoencoder, attention
and classifier-free guidance). Secondly, we systematically compare the methods
of these studies in two directions, T2I generation and T2I editing, including
the encoders and the key technologies they use. In addition, we also compare
the performance of these researches side by side in terms of datasets,
evaluation metrics, training resources, and inference speed. In addition to the
four foundation models, we survey other works on T2I, such as energy-based
models and recent Mamba and multimodality. We also investigate the potential
social impact of T2I and provide some solutions. Finally, we propose unique
insights of improving the performance of T2I models and possible future
development directions. In summary, this survey is the first systematic and
comprehensive overview of T2I, aiming to provide a valuable guide for future
researchers and stimulate continued progress in this field.

### Computers and Society

### 1. From Course to Skill: Evaluating LLM Performance in Curricular Analytics

[From Course to Skill: Evaluating LLM Performance in Curricular Analytics](http://arxiv.org/pdf/2505.02324v1)

Authors: Zhen Xu, Xinjin Li, Yingqi Huan, Veronica Minaya, Renzhe Yu

Curricular analytics (CA) -- systematic analysis of curricula data to inform
program and course refinement -- becomes an increasingly valuable tool to help
institutions align academic offerings with evolving societal and economic
demands. Large language models (LLMs) are promising for handling large-scale,
unstructured curriculum data, but it remains uncertain how reliably LLMs can
perform CA tasks. In this paper, we systematically evaluate four text alignment
strategies based on LLMs or traditional NLP methods for skill extraction, a
core task in CA. Using a stratified sample of 400 curriculum documents of
different types and a human-LLM collaborative evaluation framework, we find
that retrieval-augmented generation (RAG) to be the top-performing strategy
across all types of curriculum documents, while zero-shot prompting performs
worse than traditional NLP methods in most cases. Our findings highlight the
promise of LLMs in analyzing brief and abstract curriculum documents, but also
reveal that their performance can vary significantly depending on model
selection and prompting strategies. This underscores the importance of
carefully evaluating the performance of LLM-based strategies before large-scale
deployment.

### 2. Deaf in AI: AI language technologies and the erosion of linguistic rights

[Deaf in AI: AI language technologies and the erosion of linguistic rights](http://arxiv.org/pdf/2505.02519v1)

Authors: Maartje De Meulder

This paper explores the interplay of AI language technologies, sign language
interpreting, and linguistic access, highlighting the complex interdependencies
shaping access frameworks and the tradeoffs these technologies bring. While AI
tools promise innovation, they also perpetuate biases, reinforce technoableism,
and deepen inequalities through systemic and design flaws. The historical and
contemporary privileging of sign language interpreting as the dominant access
model, and the broader inclusion ideologies it reflects, shape AIs development
and deployment, often sidelining deaf languaging practices and introducing new
forms of linguistic subordination to technology. Drawing on Deaf Studies, Sign
Language Interpreting Studies, and crip technoscience, this paper critiques the
framing of AI as a substitute for interpreters and examines its implications
for access hierarchies. It calls for deaf-led approaches to foster AI systems
that remain equitable, inclusive, and trustworthy, supporting rather than
undermining linguistic autonomy and contributing to deaf aligned futures.

### 3. How May U.S. Courts Scrutinize Their Recidivism Risk Assessment Tools? Contextualizing AI Fairness Criteria on a Judicial Scrutiny-based Framework

[How May U.S. Courts Scrutinize Their Recidivism Risk Assessment Tools? Contextualizing AI Fairness Criteria on a Judicial Scrutiny-based Framework](http://arxiv.org/pdf/2505.02749v1)

Authors: Tin Nguyen, Jiannan Xu, Phuong-Anh Nguyen-Le, Jonathan Lazar, Donald Braman, Hal Daumé III, Zubin Jelveh

The AI/HCI and legal communities have developed largely independent
conceptualizations of fairness. This conceptual difference hinders the
potential incorporation of technical fairness criteria (e.g., procedural,
group, and individual fairness) into sustainable policies and designs,
particularly for high-stakes applications like recidivism risk assessment. To
foster common ground, we conduct legal research to identify if and how
technical AI conceptualizations of fairness surface in primary legal sources.
We find that while major technical fairness criteria can be linked to
constitutional mandates such as ``Due Process'' and ``Equal Protection'' thanks
to judicial interpretation, several challenges arise when operationalizing them
into concrete statutes/regulations. These policies often adopt procedural and
group fairness but ignore the major technical criterion of individual fairness.
Regarding procedural fairness, judicial ``scrutiny'' categories are relevant
but may not fully capture how courts scrutinize the use of demographic features
in potentially discriminatory government tools like RRA. Furthermore, some
policies contradict each other on whether to apply procedural fairness to
certain demographic features. Thus, we propose a new framework, integrating
demographics-related legal scrutiny concepts and technical fairness criteria.

### 4. Teaching the social media generation: rethinking learning without sacrificing quality

[Teaching the social media generation: rethinking learning without sacrificing quality](http://arxiv.org/pdf/2505.02770v1)

Authors: Sepinoud Azimi

The rise of social media and AI tools has reshaped how students engage with
learning, process information, and build trust in educational content. This
generation prefers short, visual materials and fast feedback but often
struggles with focus, critical thinking, and deep learning. Educators face the
challenge of adapting teaching methods to these habits without lowering
academic standards. This study presents a blended learning redesign of a
first-year technical course at a Dutch university. Key features included short
whiteboard videos before class, hands-on teamwork during class, narrative-style
handouts to reinforce learning, in-class draft assignments without AI, and
weekly anonymous feedback to adjust in real time. The results were promising:
attendance increased by nearly 50%, and none of the regularly attending
students failed the exam. Students found the videos useful but emphasized that
in-person sessions were essential for understanding the material. While some
resisted the shift in expectations, most appreciated the structure, clarity,
and opportunities for active learning. This case suggests that combining
digital familiarity with clear expectations and active support can help meet
students where they are, while still challenging them to grow.

### 5. Scoring the European Citizen in the AI Era

[Scoring the European Citizen in the AI Era](http://arxiv.org/pdf/2505.02791v1)

Authors: Nathan Genicot

Social scoring is one of the AI practices banned by the AI Act. This ban is
explicitly inspired by China, which in 2014 announced its intention to set up a
large-scale government project - the Social Credit System - aiming to rate
every Chinese citizen according to their good behaviour, using digital
technologies and AI. But in Europe, individuals are also scored by public and
private bodies in a variety of contexts, such as assessing creditworthiness,
monitoring employee productivity, detecting social fraud or terrorist risks,
and so on. However, the AI Act does not intend to prohibit these types of
scoring, as they would qualify as 'high-risk AI systems', which are authorised
while subject to various requirements. One might therefore think that the ban
on social scoring will have no practical effect on the scoring practices
already in use in Europe, and that it is merely a vague safeguard in case an
authoritarian power is tempted to set up such a system on European territory.
Contrary to this view, this article argues that the ban has been drafted in a
way that is flexible and therefore likely to make it a useful tool, similar and
complementary to Article 22 of the General Data Protection Regulation, to
protect individuals against certain forms of disproportionate use of AI-based
scoring.

### 6. What Is AI Safety? What Do We Want It to Be?

[What Is AI Safety? What Do We Want It to Be?](http://arxiv.org/pdf/2505.02313v1)

Authors: Jacqueline Harding, Cameron Domenico Kirk-Giannini

The field of AI safety seeks to prevent or reduce the harms caused by AI
systems. A simple and appealing account of what is distinctive of AI safety as
a field holds that this feature is constitutive: a research project falls
within the purview of AI safety just in case it aims to prevent or reduce the
harms caused by AI systems. Call this appealingly simple account The Safety
Conception of AI safety. Despite its simplicity and appeal, we argue that The
Safety Conception is in tension with at least two trends in the ways AI safety
researchers and organizations think and talk about AI safety: first, a tendency
to characterize the goal of AI safety research in terms of catastrophic risks
from future systems; second, the increasingly popular idea that AI safety can
be thought of as a branch of safety engineering. Adopting the methodology of
conceptual engineering, we argue that these trends are unfortunate: when we
consider what concept of AI safety it would be best to have, there are
compelling reasons to think that The Safety Conception is the answer.
Descriptively, The Safety Conception allows us to see how work on topics that
have historically been treated as central to the field of AI safety is
continuous with work on topics that have historically been treated as more
marginal, like bias, misinformation, and privacy. Normatively, taking The
Safety Conception seriously means approaching all efforts to prevent or
mitigate harms from AI systems based on their merits rather than drawing
arbitrary distinctions between them.

### 7. The use of Artificial Intelligence for Intervention and Assessment in Individuals with ASD

[The use of Artificial Intelligence for Intervention and Assessment in Individuals with ASD](http://arxiv.org/pdf/2505.02747v1)

Authors: Aggeliki Sideraki, Christos-Nikolaos Anagnostopoulos

This paper explores the use of Artificial Intelligence (AI) as a tool for
diagnosis, assessment, and intervention for individuals with Autism Spectrum
Disorder (ASD). It focuses particularly on AI's role in early diagnosis,
utilizing advanced machine learning techniques and data analysis. Recent
studies demonstrate that deep learning algorithms can identify behavioral
patterns through biometric data analysis, video-based interaction assessments,
and linguistic feature extraction, providing a more accurate and timely
diagnosis compared to traditional methods. Additionally, AI automates
diagnostic tools, reducing subjective biases and enabling the development of
personalized assessment protocols for ASD monitoring. At the same time, the
paper examines AI-powered intervention technologies, emphasizing educational
robots and adaptive communication tools. Social robotic assistants, such as NAO
and Kaspar, have been shown to enhance social skills in children by offering
structured, repetitive interactions that reinforce learning. Furthermore,
AI-driven Augmentative and Alternative Communication (AAC) systems allow
children with ASD to express themselves more effectively, while
machine-learning chatbots provide language development support through
personalized responses. The study presents research findings supporting the
effectiveness of these AI applications while addressing challenges such as
long-term evaluation and customization to individual needs. In conclusion, the
paper highlights the significance of AI as an innovative tool in ASD diagnosis
and intervention, advocating for further research to assess its long-term
impact.

### 8. Regulating Algorithmic Management: A Multi-Stakeholder Study of Challenges in Aligning Software and the Law for Workplace Scheduling

[Regulating Algorithmic Management: A Multi-Stakeholder Study of Challenges in Aligning Software and the Law for Workplace Scheduling](http://arxiv.org/pdf/2505.02329v1)

Authors: Jonathan Lynn, Rachel Y. Kim, Sicun Gao, Daniel Schneider, Sachin S. Pandya, Min Kyung Lee

The impacts of algorithmic management (AM) on worker well-being have led to
increasing calls to regulate AM practices to prevent further worker harms. Yet
existing work in aligning software with the law reduces compliance to just one
piece of the entire process of regulating AM -- which involves rule
operationalization, software use, and enforcement. We interviewed key
stakeholders involved in enforcing or complying with workplace scheduling law
-- regulators, advocates, defense attorneys, scheduling managers, and workers
($N = 38$). Based on their beliefs and experiences, we describe how scheduling
software affects beliefs about and compliance with workplace scheduling law. In
so doing, we discuss the challenges and opportunities in designing software as
a tool for regulating AM.

### 9. Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions

[Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions](http://arxiv.org/pdf/2505.02455v1)

Authors: Herminio García-González, Mike Bryant, Suzanne Swartz, Fabio Rovigo, Veerle Vanden Daelen

Historical study of the Holocaust is commonly hampered by the dispersed and
fragmented nature of important archival sources relating to this event. The
EHRI project set out to mitigate this problem by building a trans-national
network of archives, researchers, and digital practitioners, and one of its
main outcomes was the creation of the EHRI Portal, a "virtual observatory" that
gathers in one centralised platform descriptions of Holocaust-related archival
sources from around the world. In order to build the Portal a strong data
identification and integration effort was required, culminating in the
project's third phase with the creation of the EHRI-3 data integration lab. The
focus of the lab was to lower the bar to participation in the EHRI Portal by
providing support to institutions in conforming their archival metadata with
that required for integration, ultimately opening the process up to smaller
institutions (and even so-called "micro-archives") without the necessary
resources to undertake this process themselves. In this paper we present our
experiences from running the data integration lab and discuss some of the
challenges (both of a technical and social nature), how we tried to overcome
them, and the overall lessons learnt. We envisage this work as an archetype
upon which other practitioners seeking to pursue similar data integration
activities can build their own efforts.

### 10. Study of the influence of a biased database on the prediction of standard algorithms for selecting the best candidate for an interview

[Study of the influence of a biased database on the prediction of standard algorithms for selecting the best candidate for an interview](http://arxiv.org/pdf/2505.02609v1)

Authors: Shuyu Wang, Angélique Saillet, Philomène Le Gall, Alain Lacroux, Christelle Martin-Lacroux, Vincent Brault

Artificial intelligence is used at various stages of the recruitment process
to automatically select the best candidate for a position, with companies
guaranteeing unbiased recruitment. However, the algorithms used are either
trained by humans or are based on learning from past experiences that were
biased. In this article, we propose to generate data mimicking external
(discrimination) and internal biases (self-censorship) in order to train five
classic algorithms and to study the extent to which they do or do not find the
best candidates according to objective criteria. In addition, we study the
influence of the anonymisation of files on the quality of predictions.

### Databases

### 1. Wii: Dynamic Budget Reallocation In Index Tuning

[Wii: Dynamic Budget Reallocation In Index Tuning](http://arxiv.org/pdf/2505.02312v1)

Authors: Xiaoying Wang, Wentao Wu, Chi Wang, Vivek Narasayya, Surajit Chaudhuri

Index tuning aims to find the optimal index configuration for an input
workload. It is often a time-consuming and resource-intensive process, largely
attributed to the huge amount of "what-if" calls made to the query optimizer
during configuration enumeration. Therefore, in practice it is desirable to set
a budget constraint that limits the number of what-if calls allowed. This
yields a new problem of budget allocation, namely, deciding on which
query-configuration pairs (QCPs) to issue what-if calls. Unfortunately, optimal
budget allocation is NP-hard, and budget allocation decisions made by existing
solutions can be inferior. In particular, many of the what-if calls allocated
by using existing solutions are devoted to QCPs whose what-if costs can be
approximated by using cost derivation, a well-known technique that is
computationally much more efficient and has been adopted by commercial index
tuning software. This results in considerable waste of the budget, as these
what-if calls are unnecessary. In this paper, we propose "Wii," a lightweight
mechanism that aims to avoid such spurious what-if calls. It can be seamlessly
integrated with existing configuration enumeration algorithms. Experimental
evaluation on top of both standard industrial benchmarks and real workloads
demonstrates that Wii can eliminate significant number of spurious what-if
calls. Moreover, by reallocating the saved budget to QCPs where cost derivation
is less accurate, existing algorithms can be significantly improved in terms of
the final configuration found.

### 2. Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions

[Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions](http://arxiv.org/pdf/2505.02455v1)

Authors: Herminio García-González, Mike Bryant, Suzanne Swartz, Fabio Rovigo, Veerle Vanden Daelen

Historical study of the Holocaust is commonly hampered by the dispersed and
fragmented nature of important archival sources relating to this event. The
EHRI project set out to mitigate this problem by building a trans-national
network of archives, researchers, and digital practitioners, and one of its
main outcomes was the creation of the EHRI Portal, a "virtual observatory" that
gathers in one centralised platform descriptions of Holocaust-related archival
sources from around the world. In order to build the Portal a strong data
identification and integration effort was required, culminating in the
project's third phase with the creation of the EHRI-3 data integration lab. The
focus of the lab was to lower the bar to participation in the EHRI Portal by
providing support to institutions in conforming their archival metadata with
that required for integration, ultimately opening the process up to smaller
institutions (and even so-called "micro-archives") without the necessary
resources to undertake this process themselves. In this paper we present our
experiences from running the data integration lab and discuss some of the
challenges (both of a technical and social nature), how we tried to overcome
them, and the overall lessons learnt. We envisage this work as an archetype
upon which other practitioners seeking to pursue similar data integration
activities can build their own efforts.

### 3. Rethinking Federated Graph Learning: A Data Condensation Perspective

[Rethinking Federated Graph Learning: A Data Condensation Perspective](http://arxiv.org/pdf/2505.02573v1)

Authors: Hao Zhang, Xunkai Li, Yinlin Zhu, Lianglin Hu

Federated graph learning is a widely recognized technique that promotes
collaborative training of graph neural networks (GNNs) by multi-client
graphs.However, existing approaches heavily rely on the communication of model
parameters or gradients for federated optimization and fail to adequately
address the data heterogeneity introduced by intricate and diverse graph
distributions. Although some methods attempt to share additional messages among
the server and clients to improve federated convergence during communication,
they introduce significant privacy risks and increase communication overhead.
To address these issues, we introduce the concept of a condensed graph as a
novel optimization carrier to address FGL data heterogeneity and propose a new
FGL paradigm called FedGM. Specifically, we utilize a generalized condensation
graph consensus to aggregate comprehensive knowledge from distributed graphs,
while minimizing communication costs and privacy risks through a single
transmission of the condensed data. Extensive experiments on six public
datasets consistently demonstrate the superiority of FedGM over
state-of-the-art baselines, highlighting its potential for a novel FGL
paradigm.

### 4. Knowledge Graphs for Enhancing Large Language Models in Entity Disambiguation

[Knowledge Graphs for Enhancing Large Language Models in Entity Disambiguation](http://arxiv.org/pdf/2505.02737v1)

Authors: Pons Gerard, Bilalli Besim, Queralt Anna

Recent advances in Large Language Models (LLMs) have positioned them as a
prominent solution for Natural Language Processing tasks. Notably, they can
approach these problems in a zero or few-shot manner, thereby eliminating the
need for training or fine-tuning task-specific models. However, LLMs face some
challenges, including hallucination and the presence of outdated knowledge or
missing information from specific domains in the training data. These problems
cannot be easily solved by retraining the models with new data as it is a
time-consuming and expensive process. To mitigate these issues, Knowledge
Graphs (KGs) have been proposed as a structured external source of information
to enrich LLMs. With this idea, in this work we use KGs to enhance LLMs for
zero-shot Entity Disambiguation (ED). For that purpose, we leverage the
hierarchical representation of the entities' classes in a KG to gradually prune
the candidate space as well as the entities' descriptions to enrich the input
prompt with additional factual knowledge. Our evaluation on popular ED datasets
shows that the proposed method outperforms non-enhanced and description-only
enhanced LLMs, and has a higher degree of adaptability than task-specific
models. Furthermore, we conduct an error analysis and discuss the impact of the
leveraged KG's semantic expressivity on the ED performance.

### Distributed, Parallel, and Cluster Computing

### 1. Opt-GPTQ: An Optimized GPTQ Combining Sparse Attention and Quantization Techniques

[Opt-GPTQ: An Optimized GPTQ Combining Sparse Attention and Quantization Techniques](http://arxiv.org/pdf/2505.02351v1)

Authors: Jie Kong, Junxiang Zhang, Jiheng Xu, Yalong Li, Shouhua Zhang, Jiehan Zhou, Yuhai Liu, Peng Liang, Quan Zhang, Luohan Jiang

In the field of deep learning, traditional attention mechanisms face
significant challenges related to high computational complexity and large
memory consumption when processing long sequence data. To address these
limitations, we propose Opt-GPTQ, an optimized Gradient-based Post Training
Quantization (GPTQ) combining the Grouped Query Attention (GQA) mechanism with
paging memory management, optimizing the traditional Multi-Head Attention (MHA)
mechanism by grouping query heads and sharing key-value vectors. Optimized GQA
(Opt-GQA) effectively reduces computational complexity, minimizes memory
fragmentation, and enhances memory utilization for large-scale models. Opt-GPTQ
is optimized for Data Center Units (DCUs) and integrated into the vLLM model to
maximize hardware efficiency. It customizes GPU kernels to further enhance
attention computation by reducing memory access latency and boosting parallel
computing capabilities. Opt-GQA integrates Attention with Linear Biases (ALiBi)
to reduce overhead and enhance long-sequence processing. Experimental results
show that Opt?GPTQ significantly reduces computation time and memory usage
while improving model performance.

### 2. Model Checking and Synthesis for Optimal Use of Knowledge in Consensus Protocols

[Model Checking and Synthesis for Optimal Use of Knowledge in Consensus Protocols](http://arxiv.org/pdf/2505.02353v1)

Authors: Kaya Alpturer, Gerald Huang, Ron van der Meyden

Logics of knowledge and knowledge-based programs provide a way to give
abstract descriptions of solutions to problems in fault-tolerant distributed
computing, and have been used to derive optimal protocols for these problems
with respect to a variety of failure models. Generally, these results have
involved complex pencil and paper analyses with respect to the theoretical
"full-information protocol" model of information exchange between network
nodes. It is equally of interest to be able to establish the optimality of
protocols using weaker, but more practical, models of information exchange, or
else identify opportunities to improve their performance. Over the last 20
years, automated verification and synthesis tools for the logic of knowledge
have been developed, such as the model checker MCK, that can be applied to this
problem. This paper concerns the application of MCK to automated analyses of
this kind. A number of information-exchange models are considered, for
Simultaneous and Eventual variants of Byzantine Agreement under a range of
failure types. MCK is used to automatically analyze these models. The results
demonstrate that it is possible to automatically identify optimization
opportunities, and to automatically synthesize optimal protocols. The paper
provides performance measurements for the automated analysis, establishing a
benchmark for epistemic model checking and synthesis tools.

### 3. Tight Bounds on Channel Reliability via Generalized Quorum Systems (Extended Version)

[Tight Bounds on Channel Reliability via Generalized Quorum Systems (Extended Version)](http://arxiv.org/pdf/2505.02646v1)

Authors: Alejandro Naser-Pastoriza, Gregory Chockler, Alexey Gotsman, Fedor Ryabinin

Communication channel failures are a major concern for the developers of
modern fault-tolerant systems. However, while tight bounds for process failures
are well-established, extending them to include channel failures has remained
an open problem. We introduce \emph{generalized quorum systems} - a framework
that characterizes the necessary and sufficient conditions for implementing
atomic registers, atomic snapshots, lattice agreement and consensus under
arbitrary patterns of process-channel failures. Generalized quorum systems
relax the connectivity constraints of classical quorum systems: instead of
requiring bidirectional reachability for every pair of write and read quorums,
they only require some write quorum to be \emph{unidirectionally} reachable
from some read quorum. This weak connectivity makes implementing registers
particularly challenging, because it precludes the traditional request/response
pattern of quorum access, making classical solutions like ABD inapplicable. To
address this, we introduce novel logical clocks that allow write and read
quorums to reliably track state updates without relying on bi-directional
connectivity.

### 4. Optimistic, Signature-Free Reliable Broadcast and Its Applications

[Optimistic, Signature-Free Reliable Broadcast and Its Applications](http://arxiv.org/pdf/2505.02761v1)

Authors: Nibesh Shrestha, Qianyu Yu, Aniket Kate, Giuliano Losa, Kartik Nayak, Xuechao Wang

Reliable broadcast (RBC) is a key primitive in fault-tolerant distributed
systems, and improving its efficiency can benefit a wide range of applications.
This work focuses on signature-free RBC protocols, which are particularly
attractive due to their computational efficiency. Existing protocols in this
setting incur an optimal 3 steps to reach a decision while tolerating up to $f
< n/3$ Byzantine faults, where $n$ is the number of parties. In this work, we
propose an optimistic RBC protocol that maintains the $f < n/3$ fault tolerance
but achieves termination in just 2 steps under certain optimistic
conditions--when at least $\lceil \frac{n+2f-2}{2} \rceil$ non-broadcaster
parties behave honestly. We also prove a matching lower bound on the number of
honest parties required for 2-step termination.
  We show that our latency-reduction technique generalizes beyond RBC and
applies to other primitives such as asynchronous verifiable secret sharing
(AVSS) and asynchronous verifiable information dispersal (AVID), enabling them
to complete in 2 steps under similar optimistic conditions.
  To highlight the practical impact of our RBC protocol, we integrate it into
Sailfish++, a new signature-free, post-quantum secure DAG-based Byzantine
fault-tolerant (BFT) consensus protocol. Under optimistic conditions, this
protocol achieves a commit latency of 3 steps--matching the performance of the
best signature-based protocols. Our experimental evaluation shows that our
protocol significantly outperforms existing post-quantum secure and
signature-based protocols, even on machines with limited CPU resources. In
contrast, signature-based protocols require high CPU capacity to achieve
comparable performance. We have open-sourced our Rust implementation of
Sailfish++ to facilitate reproducible results.

### 5. An Almost Tight Lower Bound for Plurality Consensus with Undecided State Dynamics in the Population Protocol Model

[An Almost Tight Lower Bound for Plurality Consensus with Undecided State Dynamics in the Population Protocol Model](http://arxiv.org/pdf/2505.02765v1)

Authors: Antoine El-Hayek, Robert Elsässer, Stefan Schmid

We revisit the majority problem in the population protocol communication
model, as first studied by Angluin et al. (Distributed Computing 2008). We
consider a more general version of this problem known as plurality consensus,
which has already been studied intensively in the literature. In this problem,
each node in a system of $n$ nodes, has initially one of $k$ different
opinions, and they need to agree on the (relative) majority opinion. In
particular, we consider the important and intensively studied model of
Undecided State Dynamics.
  Our main contribution is an almost tight lower bound on the stabilization
time: we prove that there exists an initial configuration, even with bias
$\Delta = \omega(\sqrt{n\log n})$, where stabilization requires $\Omega(kn\log
\frac {\sqrt n} {k \log n})$ interactions, or equivalently, $\Omega(k\log \frac
{\sqrt n} {k \log n})$ parallel time for any $k = o\left(\frac {\sqrt n}{\log
n}\right)$. This bound is tight for any $ k \le n^{\frac 1 2 - \epsilon}$,
where $\epsilon >0$ can be any small constant, as Amir et al.~(PODC'23) gave a
$O(k\log n)$ parallel time upper bound for $k = O\left(\frac {\sqrt n} {\log ^2
n}\right)$.

### 6. Brief Announcement: Minimizing Energy Solves Relative Majority with a Cubic Number of States in Population Protocols

[Brief Announcement: Minimizing Energy Solves Relative Majority with a Cubic Number of States in Population Protocols](http://arxiv.org/pdf/2505.02785v1)

Authors: Tom-Lukas Breitkopf, Julien Dallot, Antoine El-Hayek, Stefan Schmid

This paper revisits a fundamental distributed computing problem in the
population protocol model.
  Provided $n$ agents each starting with an input color in $[k]$, the relative
majority problem asks to find the predominant color.
  In the population protocol model, at each time step, a scheduler selects two
agents that first learn each other's states and then update their states based
on what they learned.
  We present the \textsc{Circles} protocol that solves the relative majority
problem with $k^3$ states. It is always-correct under weakly fair scheduling.
  Not only does it improve upon the best known upper bound of $O(k^7)$, but it
also shows a strikingly simpler design inspired by energy minimization in
chemical settings.

### 7. Recolorable Graph Exploration by an Oblivious Agent with Fewer Colors

[Recolorable Graph Exploration by an Oblivious Agent with Fewer Colors](http://arxiv.org/pdf/2505.02789v1)

Authors: Shota Takahashi, Haruki Kanaya, Shoma Hiraoka, Ryota Eguchi, Yuichi Sudo

Recently, B\"ockenhauer, Frei, Unger, and Wehner (SIROCCO 2023) introduced a
novel variant of the graph exploration problem in which a single memoryless
agent must visit all nodes of an unknown, undirected, and connected graph
before returning to its starting node. Unlike the standard model for mobile
agents, edges are not labeled with port numbers. Instead, the agent can color
its current node and observe the color of each neighboring node. To move, it
specifies a target color and then moves to an adversarially chosen neighbor of
that color. B\"ockenhauer~et al.~analyzed the minimum number of colors required
for successful exploration and proposed an elegant algorithm that enables the
agent to explore an arbitrary graph using only eight colors. In this paper, we
present a novel graph exploration algorithm that requires only six colors.
Furthermore, we prove that five colors are sufficient if we consider only a
restricted class of graphs, which we call the $\varphi$-free graphs, a class
that includes every graph with maximum degree at most three and every cactus.

### 8. Large Language Model Partitioning for Low-Latency Inference at the Edge

[Large Language Model Partitioning for Low-Latency Inference at the Edge](http://arxiv.org/pdf/2505.02533v1)

Authors: Dimitrios Kafetzis, Ramin Khalili, Iordanis Koutsopoulos

Large Language Models (LLMs) based on autoregressive, decoder-only
Transformers generate text one token at a time, where a token represents a
discrete unit of text. As each newly produced token is appended to the partial
output sequence, the length grows and so does the memory and compute load, due
to the expanding key-value caches, which store intermediate representations of
all previously generated tokens in the multi-head attention (MHA) layer. As
this iterative process steadily increases memory and compute demands,
layer-based partitioning in resource-constrained edge environments often
results in memory overload or high inference latency. To address this and
reduce inference latency, we propose a resource-aware Transformer architecture
partitioning algorithm, where the partitioning decision is updated at regular
intervals during token generation. The approach is myopic in that it is based
on instantaneous information about device resource availability and network
link bandwidths. When first executed, the algorithm places blocks on devices,
and in later executions, it migrates these blocks among devices so that the sum
of migration delay and inference delay remains low. Our approach partitions the
decoder at the attention head level, co-locating each attention head with its
key-value cache and allowing dynamic migrations whenever resources become
tight. By allocating different attention heads to different devices, we exploit
parallel execution of attention heads and thus achieve substantial reductions
in inference delays. Our experiments show that in small-scale settings (3-5
devices), the proposed method achieves within 15 to 20 percent of an exact
optimal solver's latency, while in larger-scale tests it achieves notable
improvements in inference speed and memory usage compared to state-of-the-art
layer-based partitioning approaches.

### 9. A Unifying Framework to Enable Artificial Intelligence in High Performance Computing Workflows

[A Unifying Framework to Enable Artificial Intelligence in High Performance Computing Workflows](http://arxiv.org/pdf/2505.02738v1)

Authors: Jens Domke, Mohamed Wahib, Anshu Dubey, Tal Ben-Nun, Erik W. Draeger

Current trends point to a future where large-scale scientific applications
are tightly-coupled HPC/AI hybrids. Hence, we urgently need to invest in
creating a seamless, scalable framework where HPC and AI/ML can efficiently
work together and adapt to novel hardware and vendor libraries without starting
from scratch every few years. The current ecosystem and sparsely-connected
community are not sufficient to tackle these challenges, and we require a
breakthrough catalyst for science similar to what PyTorch enabled for AI.

### 10. Towards One-shot Federated Learning: Advances, Challenges, and Future Directions

[Towards One-shot Federated Learning: Advances, Challenges, and Future Directions](http://arxiv.org/pdf/2505.02426v1)

Authors: Flora Amato, Lingyu Qiu, Mohammad Tanveer, Salvatore Cuomo, Fabio Giampaolo, Francesco Piccialli

One-shot FL enables collaborative training in a single round, eliminating the
need for iterative communication, making it particularly suitable for use in
resource-constrained and privacy-sensitive applications. This survey offers a
thorough examination of One-shot FL, highlighting its distinct operational
framework compared to traditional federated approaches. One-shot FL supports
resource-limited devices by enabling single-round model aggregation while
maintaining data locality. The survey systematically categorizes existing
methodologies, emphasizing advancements in client model initialization,
aggregation techniques, and strategies for managing heterogeneous data
distributions. Furthermore, we analyze the limitations of current approaches,
particularly in terms of scalability and generalization in non-IID settings. By
analyzing cutting-edge techniques and outlining open challenges, this survey
aspires to provide a comprehensive reference for researchers and practitioners
aiming to design and implement One-shot FL systems, advancing the development
and adoption of One-shot FL solutions in a real-world, resource-constrained
scenario.

### Digital Libraries

### 1. Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions

[Running a Data Integration Lab in the Context of the EHRI Project: Challenges, Lessons Learnt and Future Directions](http://arxiv.org/pdf/2505.02455v1)

Authors: Herminio García-González, Mike Bryant, Suzanne Swartz, Fabio Rovigo, Veerle Vanden Daelen

Historical study of the Holocaust is commonly hampered by the dispersed and
fragmented nature of important archival sources relating to this event. The
EHRI project set out to mitigate this problem by building a trans-national
network of archives, researchers, and digital practitioners, and one of its
main outcomes was the creation of the EHRI Portal, a "virtual observatory" that
gathers in one centralised platform descriptions of Holocaust-related archival
sources from around the world. In order to build the Portal a strong data
identification and integration effort was required, culminating in the
project's third phase with the creation of the EHRI-3 data integration lab. The
focus of the lab was to lower the bar to participation in the EHRI Portal by
providing support to institutions in conforming their archival metadata with
that required for integration, ultimately opening the process up to smaller
institutions (and even so-called "micro-archives") without the necessary
resources to undertake this process themselves. In this paper we present our
experiences from running the data integration lab and discuss some of the
challenges (both of a technical and social nature), how we tried to overcome
them, and the overall lessons learnt. We envisage this work as an archetype
upon which other practitioners seeking to pursue similar data integration
activities can build their own efforts.

### Discrete Mathematics

### 1. Net Occurrences in Fibonacci and Thue-Morse Words

[Net Occurrences in Fibonacci and Thue-Morse Words](http://arxiv.org/pdf/2505.02307v1)

Authors: Peaker Guo, Kaisei Kishi

A net occurrence of a repeated string in a text is an occurrence with unique
left and right extensions, and the net frequency of the string is the number of
its net occurrences in the text. Originally introduced for applications in
Natural Language Processing, net frequency has recently gained attention for
its algorithmic aspects. Guo et al. [CPM 2024] and Ohlebusch et al. [SPIRE
2024] focus on its computation in the offline setting, while Guo et al. [SPIRE
2024], Inenaga [arXiv 2024], and Mieno and Inenaga [CPM 2025] tackle the online
counterpart. Mieno and Inenaga also characterize net occurrences in terms of
the minimal unique substrings of the text. Additionally, Guo et al. [CPM 2024]
initiate the study of net occurrences in Fibonacci words to establish a lower
bound on the asymptotic running time of algorithms. Although there has been
notable progress in algorithmic developments and some initial combinatorial
insights, the combinatorial aspects of net occurrences have yet to be
thoroughly examined. In this work, we make two key contributions. First, we
confirm the conjecture that each Fibonacci word contains exactly three net
occurrences. Second, we show that each Thue-Morse word contains exactly nine
net occurrences. To achieve these results, we introduce the notion of
overlapping net occurrence cover, which narrows down the candidate net
occurrences in any text. Furthermore, we provide a precise characterization of
occurrences of Fibonacci and Thue-Morse words of smaller order, offering
structural insights that may have independent interest and potential
applications in algorithm analysis and combinatorial properties of these words.

### 2. Linear colorings of graphs

[Linear colorings of graphs](http://arxiv.org/pdf/2505.02768v1)

Authors: Claire Hilaire, Matjaž Krnc, Martin Milanič, Jean-Florent Raymond

Motivated by algorithmic applications, Kun, O'Brien, Pilipczuk, and Sullivan
introduced the parameter linear chromatic number as a relaxation of treedepth
and proved that the two parameters are polynomially related. They conjectured
that treedepth could be bounded from above by twice the linear chromatic
number.
  In this paper we investigate the properties of linear chromatic number and
provide improved bounds in several graph classes.

### 3. i-QLS: Quantum-supported Algorithm for Least Squares Optimization in Non-Linear Regression

[i-QLS: Quantum-supported Algorithm for Least Squares Optimization in Non-Linear Regression](http://arxiv.org/pdf/2505.02788v1)

Authors: Supreeth Mysore Venkatesh, Antonio Macaluso, Diego Arenas, Matthias Klusch, Andreas Dengel

We propose an iterative quantum-assisted least squares (i-QLS) optimization
method that leverages quantum annealing to overcome the scalability and
precision limitations of prior quantum least squares approaches. Unlike
traditional QUBO-based formulations, which suffer from a qubit overhead due to
fixed discretization, our approach refines the solution space iteratively,
enabling exponential convergence while maintaining a constant qubit requirement
per iteration. This iterative refinement transforms the problem into an anytime
algorithm, allowing for flexible computational trade-offs. Furthermore, we
extend our framework beyond linear regression to non-linear function
approximation via spline-based modeling, demonstrating its adaptability to
complex regression tasks. We empirically validate i-QLS on the D-Wave quantum
annealer, showing that our method efficiently scales to high-dimensional
problems, achieving competitive accuracy with classical solvers while
outperforming prior quantum approaches. Experiments confirm that i-QLS enables
near-term quantum hardware to perform regression tasks with improved precision
and scalability, paving the way for practical quantum-assisted machine learning
applications.

### Data Structures and Algorithms

### 1. Unifying Laplace Mechanism with Instance Optimality in Differential Privacy

[Unifying Laplace Mechanism with Instance Optimality in Differential Privacy](http://arxiv.org/pdf/2505.02798v1)

Authors: David Durfee

We adapt the canonical Laplace mechanism, widely used in differentially
private data analysis, to achieve near instance optimality with respect to the
hardness of the underlying dataset. In particular, we construct a piecewise
Laplace distribution whereby we defy traditional assumptions and show that
Laplace noise can in fact be drawn proportional to the local sensitivity when
done in a piecewise manner. While it may initially seem counterintuitive that
this satisfies (pure) differential privacy and can be sampled, we provide both
through a simple connection to the exponential mechanism and inverse
sensitivity along with the fact that the Laplace distribution is a two-sided
exponential distribution. As a result, we prove that in the continuous setting
our \textit{piecewise Laplace mechanism} strictly dominates the inverse
sensitivity mechanism, which was previously shown to both be nearly instance
optimal and uniformly outperform the smooth sensitivity framework. Furthermore,
in the worst-case where all local sensitivities equal the global sensitivity,
our method simply reduces to a Laplace mechanism. We also complement this with
an approximate local sensitivity variant to potentially ease the computational
cost, which can also extend to higher dimensions.

### 2. Efficient Classical Algorithms for Simulating Gaussian Boson Sampling on Graphs

[Efficient Classical Algorithms for Simulating Gaussian Boson Sampling on Graphs](http://arxiv.org/pdf/2505.02445v1)

Authors: Yexin Zhang, Shuo Zhou, Xinzhao Wang, Ziruo Wang, Ziyi Yang, Rui Yang, Yecheng Xue, Tongyang Li

Gaussian Boson Sampling (GBS) is a promising candidate for demonstrating
quantum computational advantage and can be applied to solving graph-related
problems. In this work, we propose Markov chain Monte Carlo-based algorithms to
simulate GBS on undirected, unweighted graphs. Our main contribution is a
double-loop variant of Glauber dynamics, whose stationary distribution matches
the GBS distribution. We further prove that it mixes in polynomial time for
dense graphs using a refined canonical path argument. Numerically, we conduct
experiments on graphs with 256 vertices, larger than the scales in former GBS
experiments as well as classical simulations. In particular, we show that both
the single-loop and double-loop Glauber dynamics improve the performance of
original random search and simulated annealing algorithms for the max-Hafnian
and densest $k$-subgraph problems up to 10x. Overall, our approach offers both
theoretical guarantees and practical advantages for classical simulations of
GBS on graphs.

### Emerging Technologies

### 1. Beyond the model: Key differentiators in large language models and multi-agent services

[Beyond the model: Key differentiators in large language models and multi-agent services](http://arxiv.org/pdf/2505.02489v1)

Authors: Muskaan Goyal, Pranav Bhasin

With the launch of foundation models like DeepSeek, Manus AI, and Llama 4, it
has become evident that large language models (LLMs) are no longer the sole
defining factor in generative AI. As many now operate at comparable levels of
capability, the real race is not about having the biggest model but optimizing
the surrounding ecosystem, including data quality and management, computational
efficiency, latency, and evaluation frameworks. This review article delves into
these critical differentiators that ensure modern AI services are efficient and
profitable.

### 2. Open Challenges for a Production-ready Cloud Environment on top of RISC-V hardware

[Open Challenges for a Production-ready Cloud Environment on top of RISC-V hardware](http://arxiv.org/pdf/2505.02650v1)

Authors: Aaron Call, Ramon Nou, Guillem Senabre

As part of the Vitamin-V European project, we have built a prototype of a
RISC-V cluster managed by OpenStack, with the goal of realizing a functional
RISC-V cloud ecosystem. In this poster we explain the hardware and software
challenges encountered while porting some elements of OpenStack. We also
discuss the current performance gaps that challenge a performance-ready cloud
environment over such new ISA, an essential element to fulfill in order to
achieve european technological sovereignty.

### 3. Beyond the Monitor: Mixed Reality Visualization and AI for Enhanced Digital Pathology Workflow

[Beyond the Monitor: Mixed Reality Visualization and AI for Enhanced Digital Pathology Workflow](http://arxiv.org/pdf/2505.02780v1)

Authors: Jai Prakash Veerla, Partha Sai Guttikonda, Helen H. Shang, Mohammad Sadegh Nasr, Cesar Torres, Jacob M. Luber

Pathologists rely on gigapixel whole-slide images (WSIs) to diagnose diseases
like cancer, yet current digital pathology tools hinder diagnosis. The immense
scale of WSIs, often exceeding 100,000 X 100,000 pixels, clashes with the
limited views traditional monitors offer. This mismatch forces constant panning
and zooming, increasing pathologist cognitive load, causing diagnostic fatigue,
and slowing pathologists' adoption of digital methods. PathVis, our
mixed-reality visualization platform for Apple Vision Pro, addresses these
challenges. It transforms the pathologist's interaction with data, replacing
cumbersome mouse-and-monitor navigation with intuitive exploration using
natural hand gestures, eye gaze, and voice commands in an immersive workspace.
PathVis integrates AI to enhance diagnosis. An AI-driven search function
instantly retrieves and displays the top five similar patient cases
side-by-side, improving diagnostic precision and efficiency through rapid
comparison. Additionally, a multimodal conversational AI assistant offers
real-time image interpretation support and aids collaboration among
pathologists across multiple Apple devices. By merging the directness of
traditional pathology with advanced mixed-reality visualization and AI, PathVis
improves diagnostic workflows, reduces cognitive strain, and makes pathology
practice more effective and engaging. The PathVis source code and a demo video
are publicly available at: https://github.com/jaiprakash1824/Path_Vis

### 4. Privacy Risks and Preservation Methods in Explainable Artificial Intelligence: A Scoping Review

[Privacy Risks and Preservation Methods in Explainable Artificial Intelligence: A Scoping Review](http://arxiv.org/pdf/2505.02828v1)

Authors: Sonal Allana, Mohan Kankanhalli, Rozita Dara

Explainable Artificial Intelligence (XAI) has emerged as a pillar of
Trustworthy AI and aims to bring transparency in complex models that are opaque
by nature. Despite the benefits of incorporating explanations in models, an
urgent need is found in addressing the privacy concerns of providing this
additional information to end users. In this article, we conduct a scoping
review of existing literature to elicit details on the conflict between privacy
and explainability. Using the standard methodology for scoping review, we
extracted 57 articles from 1,943 studies published from January 2019 to
December 2024. The review addresses 3 research questions to present readers
with more understanding of the topic: (1) what are the privacy risks of
releasing explanations in AI systems? (2) what current methods have researchers
employed to achieve privacy preservation in XAI systems? (3) what constitutes a
privacy preserving explanation? Based on the knowledge synthesized from the
selected studies, we categorize the privacy risks and preservation methods in
XAI and propose the characteristics of privacy preserving explanations to aid
researchers and practitioners in understanding the requirements of XAI that is
privacy compliant. Lastly, we identify the challenges in balancing privacy with
other system desiderata and provide recommendations for achieving privacy
preserving XAI. We expect that this review will shed light on the complex
relationship of privacy and explainability, both being the fundamental
principles of Trustworthy AI.

### Graphics

### 1. GarmentImage: Raster Encoding of Garment Sewing Patterns with Diverse Topologies

[GarmentImage: Raster Encoding of Garment Sewing Patterns with Diverse Topologies](http://arxiv.org/pdf/2505.02592v1)

Authors: Yuki Tatsukawa, Anran Qi, I-Chao Shen, Takeo Igarashi

Garment sewing patterns are the design language behind clothing, yet their
current vector-based digital representations weren't built with machine
learning in mind. Vector-based representation encodes a sewing pattern as a
discrete set of panels, each defined as a sequence of lines and curves,
stitching information between panels and the placement of each panel around a
body. However, this representation causes two major challenges for neural
networks: discontinuity in latent space between patterns with different
topologies and limited generalization to garments with unseen topologies in the
training data. In this work, we introduce GarmentImage, a unified raster-based
sewing pattern representation. GarmentImage encodes a garment sewing pattern's
geometry, topology and placement into multi-channel regular grids. Machine
learning models trained on GarmentImage achieve seamless transitions between
patterns with different topologies and show better generalization capabilities
compared to models trained on vector-based representation. We demonstrate the
effectiveness of GarmentImage across three applications: pattern exploration in
latent space, text-based pattern editing, and image-to-pattern prediction. The
results show that GarmentImage achieves superior performance on these
applications using only simple convolutional networks.

### 2. Sparse Ellipsoidal Radial Basis Function Network for Point Cloud Surface Representation

[Sparse Ellipsoidal Radial Basis Function Network for Point Cloud Surface Representation](http://arxiv.org/pdf/2505.02350v1)

Authors: Bobo Lian, Dandan Wang, Chenjian Wu, Minxin Chen

Point cloud surface representation is a fundamental problem in computer
graphics and vision. This paper presents a machine learning approach for
approximating the signed distance function (SDF) of a point cloud using sparse
ellipsoidal radial basis function networks, enabling a compact and accurate
surface representation. Given the SDF values defined on the grid points
constructed from the point cloud, our method approximates the SDF accurately
with as few ellipsoidal radial basis functions (ERBFs) as possible, i.e.,
represent the SDF of a point cloud by sparse ERBFs. To balance sparsity and
approximation precision, a dynamic multi-objective optimization strategy is
introduced, which adaptively adds the regularization terms and jointly
optimizes the weights, centers, shapes, and orientations of ERBFs. To improve
computational efficiency, a nearest-neighbor-based data structure is employed,
restricting function calculations to points near each Gaussian kernel center.
The computations for each kernel are further parallelized on CUDA, which
significantly improves the optimization speed. Additionally, a hierarchical
octree-based refinement strategy is designed for training. Specifically, the
initialization and optimization of network parameters are conducted using
coarse grid points in the octree lattice structure. Subsequently, fine lattice
points are progressively incorporated to accelerate model convergence and
enhance training efficiency. Extensive experiments on multiple benchmark
datasets demonstrate that our method outperforms previous sparse representation
approaches in terms of accuracy, robustness, and computational efficiency. The
corresponding code is publicly available at
https://github.com/lianbobo/SE-RBFNet.git.

### Computer Science and Game Theory

### 1. Stochastic Games with Limited Public Memory

[Stochastic Games with Limited Public Memory](http://arxiv.org/pdf/2505.02623v1)

Authors: Kristoffer Arnsfelt Hansen, Rasmus Ibsen-Jensen, Abraham Neyman

We study the memory resources required for near-optimal play in two-player
zero-sum stochastic games with the long-run average payoff. Although optimal
strategies may not exist in such games, near-optimal strategies always do.
  Mertens and Neyman (1981) proved that in any stochastic game, for any
$\varepsilon>0$, there exist uniform $\varepsilon$-optimal memory-based
strategies -- i.e., strategies that are $\varepsilon$-optimal in all
sufficiently long $n$-stage games -- that use at most $O(n)$ memory states
within the first $n$ stages. We improve this bound on the number of memory
states by proving that in any stochastic game, for any $\varepsilon>0$, there
exist uniform $\varepsilon$-optimal memory-based strategies that use at most
$O(\log n)$ memory states in the first $n$ stages. Moreover, we establish the
existence of uniform $\varepsilon$-optimal memory-based strategies whose memory
updating and action selection are time-independent and such that, with
probability close to 1, for all $n$, the number of memory states used up to
stage $n$ is at most $O(\log n)$.
  This result cannot be extended to strategies with bounded public memory --
even if time-dependent memory updating and action selection are allowed. This
impossibility is illustrated in the Big Match -- a well-known stochastic game
where the stage payoffs to Player 1 are 0 or 1. Although for any $\varepsilon >
0$, there exist strategies of Player 1 that guarantee a payoff {exceeding} $1/2
- \varepsilon$ in all sufficiently long $n$-stage games, we show that any
strategy of Player 1 that uses a finite public memory fails to guarantee a
payoff greater than $\varepsilon$ in any sufficiently long $n$-stage game.

### 2. Adaptive Bidding Policies for First-Price Auctions with Budget Constraints under Non-stationarity

[Adaptive Bidding Policies for First-Price Auctions with Budget Constraints under Non-stationarity](http://arxiv.org/pdf/2505.02796v1)

Authors: Yige Wang, Jiashuo Jiang

We study how a budget-constrained bidder should learn to adaptively bid in
repeated first-price auctions to maximize her cumulative payoff. This problem
arose due to an industry-wide shift from second-price auctions to first-price
auctions in display advertising recently, which renders truthful bidding (i.e.,
always bidding one's private value) no longer optimal. We propose a simple
dual-gradient-descent-based bidding policy that maintains a dual variable for
budget constraint as the bidder consumes her budget. In analysis, we consider
two settings regarding the bidder's knowledge of her private values in the
future: (i) an uninformative setting where all the distributional knowledge
(can be non-stationary) is entirely unknown to the bidder, and (ii) an
informative setting where a prediction of the budget allocation in advance. We
characterize the performance loss (or regret) relative to an optimal policy
with complete information on the stochasticity. For uninformative setting, We
show that the regret is \tilde{O}(\sqrt{T}) plus a variation term that reflects
the non-stationarity of the value distributions, and this is of optimal order.
We then show that we can get rid of the variation term with the help of the
prediction; specifically, the regret is \tilde{O}(\sqrt{T}) plus the prediction
error term in the informative setting.

### 3. Incentivizing Inclusive Contributions in Model Sharing Markets

[Incentivizing Inclusive Contributions in Model Sharing Markets](http://arxiv.org/pdf/2505.02462v1)

Authors: Enpei Zhang, Jingyi Chai, Rui Ye, Yanfeng Wang, Siheng Chen

While data plays a crucial role in training contemporary AI models, it is
acknowledged that valuable public data will be exhausted in a few years,
directing the world's attention towards the massive decentralized private data.
However, the privacy-sensitive nature of raw data and lack of incentive
mechanism prevent these valuable data from being fully exploited. Addressing
these challenges, this paper proposes inclusive and incentivized personalized
federated learning (iPFL), which incentivizes data holders with diverse
purposes to collaboratively train personalized models without revealing raw
data. iPFL constructs a model-sharing market by solving a graph-based training
optimization and incorporates an incentive mechanism based on game theory
principles. Theoretical analysis shows that iPFL adheres to two key incentive
properties: individual rationality and truthfulness. Empirical studies on
eleven AI tasks (e.g., large language models' instruction-following tasks)
demonstrate that iPFL consistently achieves the highest economic utility, and
better or comparable model performance compared to baseline methods. We
anticipate that our iPFL can serve as a valuable technique for boosting future
AI models on decentralized private data while making everyone satisfied.

### Human-Computer Interaction

### 1. Can LLM-Simulated Practice and Feedback Upskill Human Counselors? A Randomized Study with 90+ Novice Counselors

[Can LLM-Simulated Practice and Feedback Upskill Human Counselors? A Randomized Study with 90+ Novice Counselors](http://arxiv.org/pdf/2505.02428v1)

Authors: Ryan Louie, Ifdita Hasan Orney, Juan Pablo Pacheco, Raj Sanjay Shah, Emma Brunskill, Diyi Yang

Training more counselors, from clinical students to peer supporters, can help
meet the demand for accessible mental health support; however, current training
approaches remain resource-intensive and difficult to scale effectively. Large
Language Models (LLMs) offer promising solutions for growing counseling skills
training through simulated practice and automated feedback. Despite successes
in aligning LLMs with expert-counselor annotations, we do not know whether
LLM-based counseling training tools -- such as AI patients that simulate
real-world challenges and generative AI feedback with suggested alternatives
and rationales -- actually lead to improvements in novice counselor skill
development. We develop CARE, an LLM-simulated practice and feedback system,
and randomize 94 novice counselors to practice using an AI patient, either
alone or with AI feedback, measuring changes in their behavioral performance,
self-assessments, and qualitative learning takeaways. Our results show the
practice-and-feedback group improved in their use of reflections and questions
(d=0.32-0.39, p$<$0.05). In contrast, the group that practiced with an AI
patient alone did not show improvements, and in the case of empathy, actually
had worse uses across time (d=$-$0.52, p=0.001) and when compared against the
practice-and-feedback group (d=0.72, p=0.001). Participants' qualitative
self-reflections revealed key differences: the practice-and-feedback group
adopted a client-centered approach involving listening to and validating
feelings, while the practice-alone group remained solution-oriented but delayed
offering suggestions until gathering more information. Overall, these results
suggest that LLM-based training systems can promote effective skill
development, but that combining both simulated practice and structured feedback
is critical.

### 2. "Salt is the Soul of Hakka Baked Chicken": Reimagining Traditional Chinese Culinary ICH for Modern Contexts Without Losing Tradition

["Salt is the Soul of Hakka Baked Chicken": Reimagining Traditional Chinese Culinary ICH for Modern Contexts Without Losing Tradition](http://arxiv.org/pdf/2505.02542v1)

Authors: Sijia Liu, XiaoKe Zeng, Fengyihan Wu, Shu Ye, Bowen Liu, Sidney Cheung, Richard William Allen, Ray Lc

Intangible Cultural Heritage (ICH) like traditional culinary practices face
increasing pressure to adapt to globalization while maintaining their cultural
authenticity. Centuries-old traditions in Chinese cuisine are subject to rapid
changes for adaptation to contemporary tastes and dietary preferences. The
preservation of these cultural practices requires approaches that can enable
ICH practitioners to reimagine and recreate ICH for modern contexts. To address
this, we created workshops where experienced practitioners of traditional
Chinese cuisine co-created recipes using GenAI tools and realized the dishes.
We found that GenAI inspired ICH practitioners to innovate recipes based on
traditional workflows for broader audiences and adapt to modern dining
contexts. However, GenAI-inspired co-creation posed challenges in maintaining
the accuracy of original ICH workflows and preserving traditional flavors in
the culinary outcomes. This study offers implications for designing human-AI
collaborative processes for safeguarding and enhancing culinary ICH.

### 3. The Turing Test Is More Relevant Than Ever

[The Turing Test Is More Relevant Than Ever](http://arxiv.org/pdf/2505.02558v1)

Authors: Avraham Rahimov, Orel Zamler, Amos Azaria

The Turing Test, first proposed by Alan Turing in 1950, has historically
served as a benchmark for evaluating artificial intelligence (AI). However,
since the release of ELIZA in 1966, and particularly with recent advancements
in large language models (LLMs), AI has been claimed to pass the Turing Test.
Furthermore, criticism argues that the Turing Test primarily assesses deceptive
mimicry rather than genuine intelligence, prompting the continuous emergence of
alternative benchmarks. This study argues against discarding the Turing Test,
proposing instead using more refined versions of it, for example, by
interacting simultaneously with both an AI and human candidate to determine who
is who, allowing a longer interaction duration, access to the Internet and
other AIs, using experienced people as evaluators, etc.
  Through systematic experimentation using a web-based platform, we demonstrate
that richer, contextually structured testing environments significantly enhance
participants' ability to differentiate between AI and human interactions.
Namely, we show that, while an off-the-shelf LLM can pass some version of a
Turing Test, it fails to do so when faced with a more robust version. Our
findings highlight that the Turing Test remains an important and effective
method for evaluating AI, provided it continues to adapt as AI technology
advances. Additionally, the structured data gathered from these improved
interactions provides valuable insights into what humans expect from truly
intelligent AI systems.

### 4. FlyHaptics: Flying Multi-contact Haptic Interface

[FlyHaptics: Flying Multi-contact Haptic Interface](http://arxiv.org/pdf/2505.02582v1)

Authors: Luis Moreno, Miguel Altamirano Cabrera, Muhammad Haris Khan, Issatay Tokmurziyev, Yara Mahmoud, Valerii Serpiva, Dzmitry Tsetserukou

This work presents FlyHaptics, an aerial haptic interface tracked via a Vicon
optical motion capture system and built around six five-bar linkage assemblies
enclosed in a lightweight protective cage. We predefined five static tactile
patterns - each characterized by distinct combinations of linkage contact
points and vibration intensities - and evaluated them in a grounded pilot
study, where participants achieved 86.5 recognition accuracy (F(4, 35) = 1.47,
p = 0.23) with no significant differences between patterns. Complementary
flight demonstrations confirmed stable hover performance and consistent force
output under realistic operating conditions. These pilot results validate the
feasibility of drone-mounted, multi-contact haptic feedback and lay the
groundwork for future integration into fully immersive VR, teleoperation, and
remote interaction scenarios.

### 5. Exploring LLM-Powered Role and Action-Switching Pedagogical Agents for History Education in Virtual Reality

[Exploring LLM-Powered Role and Action-Switching Pedagogical Agents for History Education in Virtual Reality](http://arxiv.org/pdf/2505.02699v1)

Authors: Zihao Zhu, Ao Yu, Xin Tong, Pan Hui

Multi-role pedagogical agents can create engaging and immersive learning
experiences, helping learners better understand knowledge in history learning.
However, existing pedagogical agents often struggle with multi-role
interactions due to complex controls, limited feedback forms, and difficulty
dynamically adapting to user inputs. In this study, we developed a VR prototype
with LLM-powered adaptive role-switching and action-switching pedagogical
agents to help users learn about the history of the Pavilion of Prince Teng. A
2 x 2 between-subjects study was conducted with 84 participants to assess how
adaptive role-switching and action-switching affect participants' learning
outcomes and experiences. The results suggest that adaptive role-switching
enhances participants' perception of the pedagogical agent's trustworthiness
and expertise but may lead to inconsistent learning experiences. Adaptive
action-switching increases participants' perceived social presence, expertise,
and humanness. The study did not uncover any effects of role-switching and
action-switching on usability, learning motivation, and cognitive load. Based
on the findings, we proposed five design implications for incorporating
adaptive role-switching and action-switching into future VR history education
tools.

### 6. Generating HomeAssistant Automations Using an LLM-based Chatbot

[Generating HomeAssistant Automations Using an LLM-based Chatbot](http://arxiv.org/pdf/2505.02802v1)

Authors: Mathyas Giudici, Alessandro Sironi, Ismaele Villa, Samuele Scherini, Franca Garzotto

To combat climate change, individuals are encouraged to adopt sustainable
habits, in particular, with their household, optimizing their electrical
consumption. Conversational agents, such as Smart Home Assistants, hold promise
as effective tools for promoting sustainable practices within households. Our
research investigated the application of Large Language Models (LLM) in
enhancing smart home automation and promoting sustainable household practices,
specifically using the HomeAssistant framework. In particular, it highlights
the potential of GPT models in generating accurate automation routines. While
the LLMs showed proficiency in understanding complex commands and creating
valid JSON outputs, challenges such as syntax errors and message malformations
were noted, indicating areas for further improvement. Still, despite minimal
quantitative differences between "green" and "no green" prompts, qualitative
feedback highlighted a positive shift towards sustainability in the routines
generated with environmentally focused prompts. Then, an empirical evaluation
(N=56) demonstrated that the system was well-received and found engaging by
users compared to its traditional rule-based counterpart. Our findings
highlight the role of LLMs in advancing smart home technologies and suggest
further research to refine these models for broader, real-world applications to
support sustainable living.

### 7. SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration

[SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration](http://arxiv.org/pdf/2505.02418v1)

Authors: Qiang Sun, Tingting Bi, Sirui Li, Eun-Jung Holden, Paul Duuring, Kai Niu, Wei Liu

We present \textbf{SymbioticRAG}, a novel framework that fundamentally
reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a
bidirectional learning relationship between humans and machines. Our approach
addresses two critical challenges in current RAG systems: the inherently
human-centered nature of relevance determination and users' progression from
"unconscious incompetence" in query formulation. SymbioticRAG introduces a
two-tier solution where Level 1 enables direct human curation of retrieved
content through interactive source document exploration, while Level 2 aims to
build personalized retrieval models based on captured user interactions. We
implement Level 1 through three key components: (1)~a comprehensive document
processing pipeline with specialized models for layout detection, OCR, and
extraction of tables, formulas, and figures; (2)~an extensible retriever module
supporting multiple retrieval strategies; and (3)~an interactive interface that
facilitates both user engagement and interaction data logging. We experiment
Level 2 implementation via a retriever strategy incorporated LLM summarized
user intention from user interaction logs. To maintain high-quality data
preparation, we develop a human-on-the-loop validation interface that improves
pipeline output while advancing research in specialized extraction tasks.
Evaluation across three scenarios (literature review, geological exploration,
and education) demonstrates significant improvements in retrieval relevance and
user satisfaction compared to traditional RAG approaches. To facilitate broader
research and further advancement of SymbioticRAG Level 2 implementation, we
will make our system openly accessible to the research community.

### 8. Investigating the Impact of Personalized AI Tutors on Language Learning Performance

[Investigating the Impact of Personalized AI Tutors on Language Learning Performance](http://arxiv.org/pdf/2505.02443v1)

Authors: Simon Suh

Driven by the global shift towards online learning prompted by the COVID 19
pandemic, Artificial Intelligence has emerged as a pivotal player in the field
of education. Intelligent Tutoring Systems offer a new method of personalized
teaching, replacing the limitations of traditional teaching methods. However,
concerns arise about the ability of AI tutors to address skill development and
engagement during the learning process. In this paper, I will conduct a quasi
experiment with paired sample t test on 34 students pre and post use of AI
tutors in language learning platforms like Santa and Duolingo to examine the
relationship between students engagement, academic performance, and students
satisfaction during a personalized language learning experience.

### 9. HapticVLM: VLM-Driven Texture Recognition Aimed at Intelligent Haptic Interaction

[HapticVLM: VLM-Driven Texture Recognition Aimed at Intelligent Haptic Interaction](http://arxiv.org/pdf/2505.02569v1)

Authors: Muhammad Haris Khan, Miguel Altamirano Cabrera, Dmitrii Iarchuk, Yara Mahmoud, Daria Trinitatova, Issatay Tokmurziyev, Dzmitry Tsetserukou

This paper introduces HapticVLM, a novel multimodal system that integrates
vision-language reasoning with deep convolutional networks to enable real-time
haptic feedback. HapticVLM leverages a ConvNeXt-based material recognition
module to generate robust visual embeddings for accurate identification of
object materials, while a state-of-the-art Vision-Language Model
(Qwen2-VL-2B-Instruct) infers ambient temperature from environmental cues. The
system synthesizes tactile sensations by delivering vibrotactile feedback
through speakers and thermal cues via a Peltier module, thereby bridging the
gap between visual perception and tactile experience. Experimental evaluations
demonstrate an average recognition accuracy of 84.67% across five distinct
auditory-tactile patterns and a temperature estimation accuracy of 86.7% based
on a tolerance-based evaluation method with an 8{\deg}C margin of error across
15 scenarios. Although promising, the current study is limited by the use of a
small set of prominent patterns and a modest participant pool. Future work will
focus on expanding the range of tactile patterns and increasing user studies to
further refine and validate the system's performance. Overall, HapticVLM
presents a significant step toward context-aware, multimodal haptic interaction
with potential applications in virtual reality, and assistive technologies.

### 10. AI Standardized Patient Improves Human Conversations in Advanced Cancer Care

[AI Standardized Patient Improves Human Conversations in Advanced Cancer Care](http://arxiv.org/pdf/2505.02694v1)

Authors: Kurtis Haut, Masum Hasan, Thomas Carroll, Ronald Epstein, Taylan Sen, Ehsan Hoque

Serious illness communication (SIC) in end-of-life care faces challenges such
as emotional stress, cultural barriers, and balancing hope with honesty.
Despite its importance, one of the few available ways for clinicians to
practice SIC is with standardized patients, which is expensive, time-consuming,
and inflexible. In this paper, we present SOPHIE, an AI-powered standardized
patient simulation and automated feedback system. SOPHIE combines large
language models (LLMs), a lifelike virtual avatar, and automated, personalized
feedback based on clinical literature to provide remote, on-demand SIC
training. In a randomized control study with healthcare students and
professionals, SOPHIE users demonstrated significant improvement across three
critical SIC domains: Empathize, Be Explicit, and Empower. These results
suggest that AI-driven tools can enhance complex interpersonal communication
skills, offering scalable, accessible solutions to address a critical gap in
clinician education.

### Information Retrieval

### 1. Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality

[Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality](http://arxiv.org/pdf/2505.02466v1)

Authors: Xueguang Ma, Luyu Gao, Shengyao Zhuang, Jiaqi Samantha Zhan, Jamie Callan, Jimmy Lin

Recent advancements in large language models (LLMs) have driven interest in
billion-scale retrieval models with strong generalization across retrieval
tasks and languages. Additionally, progress in large vision-language models has
created new opportunities for multimodal retrieval. In response, we have
updated the Tevatron toolkit, introducing a unified pipeline that enables
researchers to explore retriever models at different scales, across multiple
languages, and with various modalities. This demo paper highlights the
toolkit's key features, bridging academia and industry by supporting efficient
training, inference, and evaluation of neural retrievers. We showcase a unified
dense retriever achieving strong multilingual and multimodal effectiveness, and
conduct a cross-modality zero-shot study to demonstrate its research potential.
Alongside, we release OmniEmbed, to the best of our knowledge, the first
embedding model that unifies text, image document, video, and audio retrieval,
serving as a baseline for future research.

### 2. Uncertainty in Repeated Implicit Feedback as a Measure of Reliability

[Uncertainty in Repeated Implicit Feedback as a Measure of Reliability](http://arxiv.org/pdf/2505.02492v1)

Authors: Bruno Sguerra, Viet-Anh Tran, Romain Hennequin, Manuel Moussallam

Recommender systems rely heavily on user feedback to learn effective user and
item representations. Despite their widespread adoption, limited attention has
been given to the uncertainty inherent in the feedback used to train these
systems. Both implicit and explicit feedback are prone to noise due to the
variability in human interactions, with implicit feedback being particularly
challenging. In collaborative filtering, the reliability of interaction signals
is critical, as these signals determine user and item similarities. Thus,
deriving accurate confidence measures from implicit feedback is essential for
ensuring the reliability of these signals.
  A common assumption in academia and industry is that repeated interactions
indicate stronger user interest, increasing confidence in preference estimates.
However, in domains such as music streaming, repeated consumption can shift
user preferences over time due to factors like satiation and exposure. While
literature on repeated consumption acknowledges these dynamics, they are often
overlooked when deriving confidence scores for implicit feedback.
  This paper addresses this gap by focusing on music streaming, where repeated
interactions are frequent and quantifiable. We analyze how repetition patterns
intersect with key factors influencing user interest and develop methods to
quantify the associated uncertainty. These uncertainty measures are then
integrated as consistency metrics in a recommendation task. Our empirical
results show that incorporating uncertainty into user preference models yields
more accurate and relevant recommendations. Key contributions include a
comprehensive analysis of uncertainty in repeated consumption patterns, the
release of a novel dataset, and a Bayesian model for implicit listening
feedback.

### 3. Evaluating Contrastive Feedback for Effective User Simulations

[Evaluating Contrastive Feedback for Effective User Simulations](http://arxiv.org/pdf/2505.02560v1)

Authors: Andreas Konstantin Kruff, Timo Breuer, Philipp Schaer

The use of Large Language Models (LLMs) for simulating user behavior in the
domain of Interactive Information Retrieval has recently gained significant
popularity. However, their application and capabilities remain highly debated
and understudied. This study explores whether the underlying principles of
contrastive training techniques, which have been effective for fine-tuning
LLMs, can also be applied beneficially in the area of prompt engineering for
user simulations.
  Previous research has shown that LLMs possess comprehensive world knowledge,
which can be leveraged to provide accurate estimates of relevant documents.
This study attempts to simulate a knowledge state by enhancing the model with
additional implicit contextual information gained during the simulation. This
approach enables the model to refine the scope of desired documents further.
The primary objective of this study is to analyze how different modalities of
contextual information influence the effectiveness of user simulations.
  Various user configurations were tested, where models are provided with
summaries of already judged relevant, irrelevant, or both types of documents in
a contrastive manner. The focus of this study is the assessment of the impact
of the prompting techniques on the simulated user agent performance. We hereby
lay the foundations for leveraging LLMs as part of more realistic simulated
users.

### 4. Social Biases in Knowledge Representations of Wikidata separates Global North from Global South

[Social Biases in Knowledge Representations of Wikidata separates Global North from Global South](http://arxiv.org/pdf/2505.02352v1)

Authors: Paramita Das, Sai Keerthana Karnam, Aditya Soni, Animesh Mukherjee

Knowledge Graphs have become increasingly popular due to their wide usage in
various downstream applications, including information retrieval, chatbot
development, language model construction, and many others. Link prediction (LP)
is a crucial downstream task for knowledge graphs, as it helps to address the
problem of the incompleteness of the knowledge graphs. However, previous
research has shown that knowledge graphs, often created in a (semi) automatic
manner, are not free from social biases. These biases can have harmful effects
on downstream applications, especially by leading to unfair behavior toward
minority groups. To understand this issue in detail, we develop a framework --
AuditLP -- deploying fairness metrics to identify biased outcomes in LP,
specifically how occupations are classified as either male or female-dominated
based on gender as a sensitive attribute. We have experimented with the
sensitive attribute of age and observed that occupations are categorized as
young-biased, old-biased, and age-neutral. We conduct our experiments on a
large number of knowledge triples that belong to 21 different geographies
extracted from the open-sourced knowledge graph, Wikidata. Our study shows that
the variance in the biased outcomes across geographies neatly mirrors the
socio-economic and cultural division of the world, resulting in a transparent
partition of the Global North from the Global South.

### 5. SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration

[SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration](http://arxiv.org/pdf/2505.02418v1)

Authors: Qiang Sun, Tingting Bi, Sirui Li, Eun-Jung Holden, Paul Duuring, Kai Niu, Wei Liu

We present \textbf{SymbioticRAG}, a novel framework that fundamentally
reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a
bidirectional learning relationship between humans and machines. Our approach
addresses two critical challenges in current RAG systems: the inherently
human-centered nature of relevance determination and users' progression from
"unconscious incompetence" in query formulation. SymbioticRAG introduces a
two-tier solution where Level 1 enables direct human curation of retrieved
content through interactive source document exploration, while Level 2 aims to
build personalized retrieval models based on captured user interactions. We
implement Level 1 through three key components: (1)~a comprehensive document
processing pipeline with specialized models for layout detection, OCR, and
extraction of tables, formulas, and figures; (2)~an extensible retriever module
supporting multiple retrieval strategies; and (3)~an interactive interface that
facilitates both user engagement and interaction data logging. We experiment
Level 2 implementation via a retriever strategy incorporated LLM summarized
user intention from user interaction logs. To maintain high-quality data
preparation, we develop a human-on-the-loop validation interface that improves
pipeline output while advancing research in specialized extraction tasks.
Evaluation across three scenarios (literature review, geological exploration,
and education) demonstrates significant improvements in retrieval relevance and
user satisfaction compared to traditional RAG approaches. To facilitate broader
research and further advancement of SymbioticRAG Level 2 implementation, we
will make our system openly accessible to the research community.

### 6. Predicting Movie Hits Before They Happen with LLMs

[Predicting Movie Hits Before They Happen with LLMs](http://arxiv.org/pdf/2505.02693v1)

Authors: Shaghayegh Agah, Yejin Kim, Neeraj Sharma, Mayur Nankani, Kevin Foley, H. Howie Huang, Sardar Hamidian

Addressing the cold-start issue in content recommendation remains a critical
ongoing challenge. In this work, we focus on tackling the cold-start problem
for movies on a large entertainment platform. Our primary goal is to forecast
the popularity of cold-start movies using Large Language Models (LLMs)
leveraging movie metadata. This method could be integrated into retrieval
systems within the personalization pipeline or could be adopted as a tool for
editorial teams to ensure fair promotion of potentially overlooked movies that
may be missed by traditional or algorithmic solutions. Our study validates the
effectiveness of this approach compared to established baselines and those we
developed.

### 7. Using Knowledge Graphs to harvest datasets for efficient CLIP model training

[Using Knowledge Graphs to harvest datasets for efficient CLIP model training](http://arxiv.org/pdf/2505.02746v1)

Authors: Simon Ging, Sebastian Walter, Jelena Bratulić, Johannes Dienert, Hannah Bast, Thomas Brox

Training high-quality CLIP models typically requires enormous datasets, which
limits the development of domain-specific models -- especially in areas that
even the largest CLIP models do not cover well -- and drives up training costs.
This poses challenges for scientific research that needs fine-grained control
over the training procedure of CLIP models. In this work, we show that by
employing smart web search strategies enhanced with knowledge graphs, a robust
CLIP model can be trained from scratch with considerably less data.
Specifically, we demonstrate that an expert foundation model for living
organisms can be built using just 10M images. Moreover, we introduce EntityNet,
a dataset comprising 33M images paired with 46M text descriptions, which
enables the training of a generic CLIP model in significantly reduced time.

### 8. Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing

[Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing](http://arxiv.org/pdf/2505.02811v1)

Authors: Diji Yang, Linda Zeng, Jinmeng Rao, Yi Zhang

Retrieval Augmented Generation (RAG) has shown strong capability in enhancing
language models' knowledge and reducing AI generative hallucinations, driving
its widespread use. However, complex tasks requiring multi-round retrieval
remain challenging, and early attempts tend to be overly optimistic without a
good sense of self-skepticism. Current multi-round RAG systems may continue
searching even when enough information has already been retrieved, or they may
provide incorrect answers without having sufficient information or knowledge.
Existing solutions either require large amounts of expensive human-labeled
process supervision data or lead to subpar performance.
  This paper aims to address these limitations by introducing a new framework,
\textbf{SIM-RAG}, to explicitly enhance RAG systems' self-awareness and
multi-round retrieval capabilities. To train SIM-RAG, we first let a RAG system
self-practice multi-round retrieval, augmenting existing question-answer pairs
with intermediate inner monologue reasoning steps to generate synthetic
training data. For each pair, the system may explore multiple retrieval paths,
which are labeled as successful if they reach the correct answer and
unsuccessful otherwise. Using this data, we train a lightweight information
sufficiency Critic. At inference time, the Critic evaluates whether the RAG
system has retrieved sufficient information at each round, guiding retrieval
decisions and improving system-level self-awareness through in-context
reinforcement learning.
  Experiments across multiple prominent RAG benchmarks show that SIM-RAG is an
effective multi-round RAG solution. Furthermore, this framework is
system-efficient, adding a lightweight component to RAG without requiring
modifications to existing LLMs or search engines, and data-efficient,
eliminating the need for costly human-annotated mid-step retrieval process
supervision data.

### Machine Learning

### 1. EntroLLM: Entropy Encoded Weight Compression for Efficient Large Language Model Inference on Edge Devices

[EntroLLM: Entropy Encoded Weight Compression for Efficient Large Language Model Inference on Edge Devices](http://arxiv.org/pdf/2505.02380v1)

Authors: Arnab Sanyal, Prithwish Mukherjee, Gourav Datta, Sandeep P. Chinchali

Large Language Models (LLMs) demonstrate exceptional performance across
various tasks, but their large storage and computational requirements constrain
their deployment on edge devices. To address this, we propose EntroLLM, a novel
compression framework that integrates mixed quantization with entropy coding to
reduce storage overhead while maintaining model accuracy. Our method applies a
layer-wise mixed quantization scheme - choosing between symmetric and
asymmetric quantization based on individual layer weight distributions - to
optimize compressibility. We then employ Huffman encoding for lossless
compression of the quantized weights, significantly reducing memory bandwidth
requirements. Furthermore, we introduce parallel Huffman decoding, which
enables efficient retrieval of encoded weights during inference, ensuring
minimal latency impact. Our experiments on edge-compatible LLMs, including
smolLM-1.7B-Instruct, phi3-mini-4k-Instruct, and mistral-7B-Instruct,
demonstrate that EntroLLM achieves up to $30%$ storage reduction compared to
uint8 models and up to $65%$ storage reduction compared to uint4 models, while
preserving perplexity and accuracy, on language benchmark tasks. We further
show that our method enables $31.9%$ - $146.6%$ faster inference throughput on
memory-bandwidth-limited edge devices, such as NVIDIA Jetson P3450, by reducing
the required data movement. The proposed approach requires no additional
re-training and is fully compatible with existing post-training quantization
methods, making it a practical solution for edge LLMs.

### 2. Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret

[Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret](http://arxiv.org/pdf/2505.02383v1)

Authors: Bingshan Hu, Zhiming Huang, Tianyue H. Zhang, Mathias Lécuyer, Nidhi Hegde

We address differentially private stochastic bandit problems from the angles
of exploring the deep connections among Thompson Sampling with Gaussian priors,
Gaussian mechanisms, and Gaussian differential privacy (GDP). We propose
DP-TS-UCB, a novel parametrized private bandit algorithm that enables to trade
off privacy and regret. DP-TS-UCB satisfies $ \tilde{O}
\left(T^{0.25(1-\alpha)}\right)$-GDP and enjoys an $O
\left(K\ln^{\alpha+1}(T)/\Delta \right)$ regret bound, where $\alpha \in [0,1]$
controls the trade-off between privacy and regret. Theoretically, our DP-TS-UCB
relies on anti-concentration bounds of Gaussian distributions and links
exploration mechanisms in Thompson Sampling-based algorithms and Upper
Confidence Bound-based algorithms, which may be of independent interest.

### 3. Exploring Design Choices for Autoregressive Deep Learning Climate Models

[Exploring Design Choices for Autoregressive Deep Learning Climate Models](http://arxiv.org/pdf/2505.02506v1)

Authors: Florian Gallusser, Simon Hentschel, Anna Krause, Andreas Hotho

Deep Learning models have achieved state-of-the-art performance in
medium-range weather prediction but often fail to maintain physically
consistent rollouts beyond 14 days. In contrast, a few atmospheric models
demonstrate stability over decades, though the key design choices enabling this
remain unclear. This study quantitatively compares the long-term stability of
three prominent DL-MWP architectures - FourCastNet, SFNO, and ClimaX - trained
on ERA5 reanalysis data at 5.625{\deg} resolution. We systematically assess the
impact of autoregressive training steps, model capacity, and choice of
prognostic variables, identifying configurations that enable stable 10-year
rollouts while preserving the statistical properties of the reference dataset.
Notably, rollouts with SFNO exhibit the greatest robustness to hyperparameter
choices, yet all models can experience instability depending on the random seed
and the set of prognostic variables

### 4. FedSDAF: Leveraging Source Domain Awareness for Enhanced Federated Domain Generalization

[FedSDAF: Leveraging Source Domain Awareness for Enhanced Federated Domain Generalization](http://arxiv.org/pdf/2505.02515v1)

Authors: Hongze Li, Zesheng Zhou, Zhenbiao Cao, Xinhui Li, Wei Chen, Xiaojin Zhang

Traditional domain generalization approaches predominantly focus on
leveraging target domain-aware features while overlooking the critical role of
source domain-specific characteristics, particularly in federated settings with
inherent data isolation. To address this gap, we propose the Federated Source
Domain Awareness Framework (FedSDAF), the first method to systematically
exploit source domain-aware features for enhanced federated domain
generalization (FedDG). The FedSDAF framework consists of two synergistic
components: the Domain-Invariant Adapter, which preserves critical
domain-invariant features, and the Domain-Aware Adapter, which extracts and
integrates source domain-specific knowledge using a Multihead Self-Attention
mechanism (MHSA). Furthermore, we introduce a bidirectional knowledge
distillation mechanism that fosters knowledge sharing among clients while
safeguarding privacy. Our approach represents the first systematic exploitation
of source domain-aware features, resulting in significant advancements in model
generalization capability.Extensive experiments on four standard benchmarks
(OfficeHome, PACS, VLCS, and DomainNet) show that our method consistently
surpasses state-of-the-art federated domain generalization approaches, with
accuracy gains of 5.2-13.8%. The source code is available at
https://github.com/pizzareapers/FedSDAF.

### 5. Low-Loss Space in Neural Networks is Continuous and Fully Connected

[Low-Loss Space in Neural Networks is Continuous and Fully Connected](http://arxiv.org/pdf/2505.02604v1)

Authors: Yongding Tian, Zaid Al-Ars, Maksim Kitsak, Peter Hofstee

Visualizations of the loss landscape in neural networks suggest that minima
are isolated points. However, both theoretical and empirical studies indicate
that it is possible to connect two different minima with a path consisting of
intermediate points that also have low loss. In this study, we propose a new
algorithm which investigates low-loss paths in the full parameter space, not
only between two minima. Our experiments on LeNet5, ResNet18, and Compact
Convolutional Transformer architectures consistently demonstrate the existence
of such continuous paths in the parameter space. These results suggest that the
low-loss region is a fully connected and continuous space in the parameter
space. Our findings provide theoretical insight into neural network
over-parameterization, highlighting that parameters collectively define a
high-dimensional low-loss space, implying parameter redundancy exists only
within individual models and not throughout the entire low-loss space.
Additionally, our work also provides new visualization methods and
opportunities to improve model generalization by exploring the low-loss space
that is closer to the origin.

### 6. Less is More: Efficient Weight Farcasting with 1-Layer Neural Network

[Less is More: Efficient Weight Farcasting with 1-Layer Neural Network](http://arxiv.org/pdf/2505.02714v1)

Authors: Xiao Shou, Debarun Bhattacharjya, Yanna Ding, Chen Zhao, Rui Li, Jianxi Gao

Addressing the computational challenges inherent in training large-scale deep
neural networks remains a critical endeavor in contemporary machine learning
research. While previous efforts have focused on enhancing training efficiency
through techniques such as gradient descent with momentum, learning rate
scheduling, and weight regularization, the demand for further innovation
continues to burgeon as model sizes keep expanding. In this study, we introduce
a novel framework which diverges from conventional approaches by leveraging
long-term time series forecasting techniques. Our method capitalizes solely on
initial and final weight values, offering a streamlined alternative for complex
model architectures. We also introduce a novel regularizer that is tailored to
enhance the forecasting performance of our approach. Empirical evaluations
conducted on synthetic weight sequences and real-world deep learning
architectures, including the prominent large language model DistilBERT,
demonstrate the superiority of our method in terms of forecasting accuracy and
computational efficiency. Notably, our framework showcases improved performance
while requiring minimal additional computational overhead, thus presenting a
promising avenue for accelerating the training process across diverse tasks and
architectures.

### 7. Entropy-Guided Sampling of Flat Modes in Discrete Spaces

[Entropy-Guided Sampling of Flat Modes in Discrete Spaces](http://arxiv.org/pdf/2505.02296v1)

Authors: Pinaki Mohanty, Riddhiman Bhattacharya, Ruqi Zhang

Sampling from flat modes in discrete spaces is a crucial yet underexplored
problem. Flat modes represent robust solutions and have broad applications in
combinatorial optimization and discrete generative modeling. However, existing
sampling algorithms often overlook the mode volume and struggle to capture flat
modes effectively. To address this limitation, we propose \emph{Entropic
Discrete Langevin Proposal} (EDLP), which incorporates local entropy into the
sampling process through a continuous auxiliary variable under a joint
distribution. The local entropy term guides the discrete sampler toward flat
modes with a small overhead. We provide non-asymptotic convergence guarantees
for EDLP in locally log-concave discrete distributions. Empirically, our method
consistently outperforms traditional approaches across tasks that require
sampling from flat basins, including Bernoulli distribution, restricted
Boltzmann machines, combinatorial optimization, and binary neural networks.

### 8. Adaptive Scoring and Thresholding with Human Feedback for Robust Out-of-Distribution Detection

[Adaptive Scoring and Thresholding with Human Feedback for Robust Out-of-Distribution Detection](http://arxiv.org/pdf/2505.02299v1)

Authors: Daisuke Yamada, Harit Vishwakarma, Ramya Korlakai Vinayak

Machine Learning (ML) models are trained on in-distribution (ID) data but
often encounter out-of-distribution (OOD) inputs during deployment -- posing
serious risks in safety-critical domains. Recent works have focused on
designing scoring functions to quantify OOD uncertainty, with score thresholds
typically set based solely on ID data to achieve a target true positive rate
(TPR), since OOD data is limited before deployment. However, these TPR-based
thresholds leave false positive rates (FPR) uncontrolled, often resulting in
high FPRs where OOD points are misclassified as ID. Moreover, fixed scoring
functions and thresholds lack the adaptivity needed to handle newly observed,
evolving OOD inputs, leading to sub-optimal performance. To address these
challenges, we propose a human-in-the-loop framework that \emph{safely updates
both scoring functions and thresholds on the fly} based on real-world OOD
inputs. Our method maximizes TPR while strictly controlling FPR at all times,
even as the system adapts over time. We provide theoretical guarantees for FPR
control under stationary conditions and present extensive empirical evaluations
on OpenOOD benchmarks to demonstrate that our approach outperforms existing
methods by achieving higher TPRs while maintaining FPR control.

### 9. Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training

[Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training](http://arxiv.org/pdf/2505.02360v1)

Authors: Fares B. Mehouachi, Saif Eddin Jabari

Adversarial training is a cornerstone of robust deep learning, but fast
methods like the Fast Gradient Sign Method (FGSM) often suffer from
Catastrophic Overfitting (CO), where models become robust to single-step
attacks but fail against multi-step variants. While existing solutions rely on
noise injection, regularization, or gradient clipping, we propose a novel
solution that purely controls the $l^p$ training norm to mitigate CO.
  Our study is motivated by the empirical observation that CO is more prevalent
under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we
develop a framework for generalized $l^p$ attack as a fixed point problem and
craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to
$l^{\infty}$. This leads to our core insight: CO emerges when highly
concentrated gradients where information localizes in few dimensions interact
with aggressive norm constraints. By quantifying gradient concentration through
Participation Ratio and entropy measures, we develop an adaptive $l^p$-FGSM
that automatically tunes the training norm based on gradient information.
Extensive experiments demonstrate that this approach achieves strong robustness
without requiring additional regularization or noise injection, providing a
novel and theoretically-principled pathway to mitigate the CO problem.

### 10. Quantitative Analysis of Performance Drop in DeepSeek Model Quantization

[Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](http://arxiv.org/pdf/2505.02390v1)

Authors: Enbo Zhao, Yi Shen, Shuming Shi, Jieyun Huang, Zhihao Chen, Ning Wang, Siqi Xiao, Jian Zhang, Kai Wang, Shiguo Lian

Recently, there is a high demand for deploying DeepSeek-R1 and V3 locally,
possibly because the official service often suffers from being busy and some
organizations have data privacy concerns. While single-machine deployment
offers infrastructure simplicity, the models' 671B FP8 parameter configuration
exceeds the practical memory limits of a standard 8-GPU machine. Quantization
is a widely used technique that helps reduce model memory consumption. However,
it is unclear what the performance of DeepSeek-R1 and V3 will be after being
quantized. This technical report presents the first quantitative evaluation of
multi-bitwidth quantization across the complete DeepSeek model spectrum. Key
findings reveal that 4-bit quantization maintains little performance
degradation versus FP8 while enabling single-machine deployment on standard
NVIDIA GPU devices. We further propose DQ3_K_M, a dynamic 3-bit quantization
method that significantly outperforms traditional Q3_K_M variant on various
benchmarks, which is also comparable with 4-bit quantization (Q4_K_M) approach
in most tasks. Moreover, DQ3_K_M supports single-machine deployment
configurations for both NVIDIA H100/A100 and Huawei 910B. Our implementation of
DQ3\_K\_M is released at https://github.com/UnicomAI/DeepSeek-Eval, containing
optimized 3-bit quantized variants of both DeepSeek-R1 and DeepSeek-V3.

### Neural and Evolutionary Computing

### 1. Giving Simulated Cells a Voice: Evolving Prompt-to-Intervention Models for Cellular Control

[Giving Simulated Cells a Voice: Evolving Prompt-to-Intervention Models for Cellular Control](http://arxiv.org/pdf/2505.02766v1)

Authors: Nam H. Le, Patrick Erikson, Yanbo Zhang, Michael Levin, Josh Bongard

Guiding biological systems toward desired states, such as morphogenetic
outcomes, remains a fundamental challenge with far-reaching implications for
medicine and synthetic biology. While large language models (LLMs) have enabled
natural language as an interface for interpretable control in AI systems, their
use as mediators for steering biological or cellular dynamics remains largely
unexplored.
  In this work, we present a functional pipeline that translates natural
language prompts into spatial vector fields capable of directing simulated
cellular collectives. Our approach combines a large language model with an
evolvable neural controller (Prompt-to-Intervention, or P2I), optimized via
evolutionary strategies to generate behaviors such as clustering or scattering
in a simulated 2D environment.
  We demonstrate that even with constrained vocabulary and simplified cell
models, evolved P2I networks can successfully align cellular dynamics with
user-defined goals expressed in plain language. This work offers a complete
loop from language input to simulated bioelectric-like intervention to
behavioral output, providing a foundation for future systems capable of natural
language-driven cellular control.

### 2. Sharpness-Aware Minimization with Z-Score Gradient Filtering for Neural Networks

[Sharpness-Aware Minimization with Z-Score Gradient Filtering for Neural Networks](http://arxiv.org/pdf/2505.02369v1)

Authors: Juyoung Yun

Generalizing well in deep neural networks remains a core challenge,
particularly due to their tendency to converge to sharp minima that degrade
robustness. Sharpness-Aware Minimization (SAM) mitigates this by seeking
flatter minima but perturbs parameters using the full gradient, which can
include statistically insignificant directions. We propose ZSharp, a simple yet
effective extension to SAM that applies layer-wise Z-score normalization
followed by percentile-based filtering to retain only statistically significant
gradient components. This selective perturbation aligns updates with
curvature-sensitive directions, enhancing generalization without requiring
architectural changes. ZSharp introduces only one additional hyperparameter,
the percentile threshold, and remains fully compatible with existing SAM
variants. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet using ResNet,
VGG, and Vision Transformers show that ZSharp consistently outperforms SAM and
its variants in test accuracy, particularly on deeper and transformer-based
models. These results demonstrate that ZSharp is a principled and lightweight
improvement for sharpness-aware optimization.

### Networking and Internet Architecture

### 1. Trustworthy Inter-Provider Agreements in 6G Using a Privacy-Enabled Hybrid Blockchain Framework

[Trustworthy Inter-Provider Agreements in 6G Using a Privacy-Enabled Hybrid Blockchain Framework](http://arxiv.org/pdf/2505.02513v1)

Authors: Farhana Javed, Josep Mangues-Bafalluy

Inter-provider agreements are central to 6G networks, where administrative
domains must securely and dynamically share services. To address the dual need
for transparency and confidentiality, we propose a privacy-enabled hybrid
blockchain setup using Hyperledger Besu, integrating both public and private
transaction workflows. The system enables decentralized service registration,
selection, and SLA breach reporting through role-based smart contracts and
privacy groups. We design and deploy a proof-of-concept implementation,
evaluating performance using end-to-end latency as a key metric within privacy
groups. Results show that public interactions maintain stable latency, while
private transactions incur additional overhead due to off-chain coordination.
The block production rate governed by IBFT 2.0 had limited impact on private
transaction latency, due to encryption and peer synchronization. Lessons
learned highlight design considerations for smart contract structure, validator
management, and scalability patterns suitable for dynamic inter-domain
collaboration. Our findings offer practical insights for deploying trustworthy
agreement systems in 6G networks using privacy-enabled hybrid blockchains.

### 2. Energy Efficiency Maximization for CR-NOMA based Smart Grid Communication Network

[Energy Efficiency Maximization for CR-NOMA based Smart Grid Communication Network](http://arxiv.org/pdf/2505.02530v1)

Authors: Mubashar Sarfraz, Sheraz Alam, Sajjad A. Ghauri, Asad Mahmood

Managing massive data flows effectively and resolving spectrum shortages are
two challenges that Smart Grid Communication Networks (SGCN) must overcome. To
address these problems, we provide a combined optimization approach that makes
use of Cognitive Radio (CR) and Non-Orthogonal Multiple Access (NOMA)
technologies. Our work focuses on using user pairing (UP) and power allocation
(PA) techniques to maximize energy efficiency (EE) in SGCN, particularly within
Neighbourhood Area Networks (NANs). We develop a joint optimization problem
that takes into account the real-world limitations of a CR-NOMA setting. This
problem is NP-hard, nonlinear, and nonconvex by nature. To address the
computational complexity of the problem, we use the Block Coordinate Descent
(BCD) method, which breaks the problem into UP and PA subproblems. Initially,
we proposed the Zebra-Optimization User Pairing (ZOUP) algorithm to tackle the
UP problem, which outperforms both Orthogonal Multiple Access (OMA) and
non-optimized NOMA (UPWO) by 78.8\% and 13.6\%, respectively, at a SNR of 15
dB. Based on the ZOUP pairs, we subsequently proposed the PA approach, i.e.,
ZOUPPA, which significantly outperforms UPWO and ZOUP by 53.2\% and 25.4\%,
respectively, at an SNR of 15 dB. A detailed analysis of key parameters,
including varying SNRs, power allocation constants, path loss exponents, user
density, channel availability, and coverage radius, underscores the superiority
of our approach. By facilitating the effective use of communication resources
in SGCN, our research opens the door to more intelligent and energy-efficient
grid systems. Our work tackles important issues in SGCN and lays the groundwork
for future developments in smart grid communication technologies by combining
modern optimization approaches with CR-NOMA.

### 3. Antifragility of RIS-assisted Communication Systems under Jamming Attacks

[Antifragility of RIS-assisted Communication Systems under Jamming Attacks](http://arxiv.org/pdf/2505.02565v1)

Authors: Mounir Bensalem, Thomas Röthig, Admela Jukan

Antifragility of communication systems is defined as measure of benefits
gained from the adverse events and variability of its environment. In this
paper, we introduce the notion of antifragility in Reconfigurable Intelligent
Surface (RIS) assisted communication systems affected by a jamming attack. We
analyzed the antifragility of the two hop systems, where the wireless path
contains source node, RIS, destination node, and a eavesdropping/jamming node.
We propose and analyze the antifragility performance for several jamming
models, such as Digital Radio Frequency Memory (DRFM) and phase and amplitude
shifting. Our paper shows that antifragility throughput can indeed be achieved
under certain power thresholds and for various jamming models. In particular,
high jamming power combined with low baseline data rates yields an antifragile
gain factor of approximately five times. The results confirm that
reconfigurable intelligent surfaces, when coupled with an antifragile design
philosophy, can convert hostile interference from a liability into a throughput
gain.

### 4. Adaptive Budgeted Multi-Armed Bandits for IoT with Dynamic Resource Constraints

[Adaptive Budgeted Multi-Armed Bandits for IoT with Dynamic Resource Constraints](http://arxiv.org/pdf/2505.02640v1)

Authors: Shubham Vaishnav, Praveen Kumar Donta, Sindri Magnússon

Internet of Things (IoT) systems increasingly operate in environments where
devices must respond in real time while managing fluctuating resource
constraints, including energy and bandwidth. Yet, current approaches often fall
short in addressing scenarios where operational constraints evolve over time.
To address these limitations, we propose a novel Budgeted Multi-Armed Bandit
framework tailored for IoT applications with dynamic operational limits. Our
model introduces a decaying violation budget, which permits limited constraint
violations early in the learning process and gradually enforces stricter
compliance over time. We present the Budgeted Upper Confidence Bound (UCB)
algorithm, which adaptively balances performance optimization and compliance
with time-varying constraints. We provide theoretical guarantees showing that
Budgeted UCB achieves sublinear regret and logarithmic constraint violations
over the learning horizon. Extensive simulations in a wireless communication
setting show that our approach achieves faster adaptation and better constraint
satisfaction than standard online learning methods. These results highlight the
framework's potential for building adaptive, resource-aware IoT systems.

### Robotics

### 1. Re-purposing a modular origami manipulator into an adaptive physical computer for machine learning and robotic perception

[Re-purposing a modular origami manipulator into an adaptive physical computer for machine learning and robotic perception](http://arxiv.org/pdf/2505.02744v1)

Authors: Jun Wang, Suyi Li

Physical computing has emerged as a powerful tool for performing intelligent
tasks directly in the mechanical domain of functional materials and robots,
reducing our reliance on the more traditional COMS computers. However, no
systematic study explains how mechanical design can influence physical
computing performance. This study sheds insights into this question by
repurposing an origami-inspired modular robotic manipulator into an adaptive
physical reservoir and systematically evaluating its computing capacity with
different physical configurations, input setups, and computing tasks. By
challenging this adaptive reservoir computer to complete the classical NARMA
benchmark tasks, this study shows that its time series emulation performance
directly correlates to the Peak Similarity Index (PSI), which quantifies the
frequency spectrum correlation between the target output and reservoir
dynamics. The adaptive reservoir also demonstrates perception capabilities,
accurately extracting its payload weight and orientation information from the
intrinsic dynamics. Importantly, such information extraction capability can be
measured by the spatial correlation between nodal dynamics within the reservoir
body. Finally, by integrating shape memory alloy (SMA) actuation, this study
demonstrates how to exploit such computing power embodied in the physical body
for practical, robotic operations. This study provides a strategic framework
for harvesting computing power from soft robots and functional materials,
demonstrating how design parameters and input selection can be configured based
on computing task requirements. Extending this framework to bio-inspired
adaptive materials, prosthetics, and self-adaptive soft robotic systems could
enable next-generation embodied intelligence, where the physical structure can
compute and interact with their digital counterparts.

### 2. Estimating Commonsense Scene Composition on Belief Scene Graphs

[Estimating Commonsense Scene Composition on Belief Scene Graphs](http://arxiv.org/pdf/2505.02405v1)

Authors: Mario A. V. Saucedo, Vignesh Kottayam Viswanathan, Christoforos Kanellakis, George Nikolakopoulos

This work establishes the concept of commonsense scene composition, with a
focus on extending Belief Scene Graphs by estimating the spatial distribution
of unseen objects. Specifically, the commonsense scene composition capability
refers to the understanding of the spatial relationships among related objects
in the scene, which in this article is modeled as a joint probability
distribution for all possible locations of the semantic object class. The
proposed framework includes two variants of a Correlation Information (CECI)
model for learning probability distributions: (i) a baseline approach based on
a Graph Convolutional Network, and (ii) a neuro-symbolic extension that
integrates a spatial ontology based on Large Language Models (LLMs).
Furthermore, this article provides a detailed description of the dataset
generation process for such tasks. Finally, the framework has been validated
through multiple runs on simulated data, as well as in a real-world indoor
environment, demonstrating its ability to spatially interpret scenes across
different room types.

### 3. Automated Hybrid Reward Scheduling via Large Language Models for Robotic Skill Learning

[Automated Hybrid Reward Scheduling via Large Language Models for Robotic Skill Learning](http://arxiv.org/pdf/2505.02483v1)

Authors: Changxin Huang, Junyang Liang, Yanbin Chang, Jingzhao Xu, Jianqiang Li

Enabling a high-degree-of-freedom robot to learn specific skills is a
challenging task due to the complexity of robotic dynamics. Reinforcement
learning (RL) has emerged as a promising solution; however, addressing such
problems requires the design of multiple reward functions to account for
various constraints in robotic motion. Existing approaches typically sum all
reward components indiscriminately to optimize the RL value function and
policy. We argue that this uniform inclusion of all reward components in policy
optimization is inefficient and limits the robot's learning performance. To
address this, we propose an Automated Hybrid Reward Scheduling (AHRS) framework
based on Large Language Models (LLMs). This paradigm dynamically adjusts the
learning intensity of each reward component throughout the policy optimization
process, enabling robots to acquire skills in a gradual and structured manner.
Specifically, we design a multi-branch value network, where each branch
corresponds to a distinct reward component. During policy optimization, each
branch is assigned a weight that reflects its importance, and these weights are
automatically computed based on rules designed by LLMs. The LLM generates a
rule set in advance, derived from the task description, and during training, it
selects a weight calculation rule from the library based on language prompts
that evaluate the performance of each branch. Experimental results demonstrate
that the AHRS method achieves an average 6.48% performance improvement across
multiple high-degree-of-freedom robotic tasks.

### 4. HapticVLM: VLM-Driven Texture Recognition Aimed at Intelligent Haptic Interaction

[HapticVLM: VLM-Driven Texture Recognition Aimed at Intelligent Haptic Interaction](http://arxiv.org/pdf/2505.02569v1)

Authors: Muhammad Haris Khan, Miguel Altamirano Cabrera, Dmitrii Iarchuk, Yara Mahmoud, Daria Trinitatova, Issatay Tokmurziyev, Dzmitry Tsetserukou

This paper introduces HapticVLM, a novel multimodal system that integrates
vision-language reasoning with deep convolutional networks to enable real-time
haptic feedback. HapticVLM leverages a ConvNeXt-based material recognition
module to generate robust visual embeddings for accurate identification of
object materials, while a state-of-the-art Vision-Language Model
(Qwen2-VL-2B-Instruct) infers ambient temperature from environmental cues. The
system synthesizes tactile sensations by delivering vibrotactile feedback
through speakers and thermal cues via a Peltier module, thereby bridging the
gap between visual perception and tactile experience. Experimental evaluations
demonstrate an average recognition accuracy of 84.67% across five distinct
auditory-tactile patterns and a temperature estimation accuracy of 86.7% based
on a tolerance-based evaluation method with an 8{\deg}C margin of error across
15 scenarios. Although promising, the current study is limited by the use of a
small set of prominent patterns and a modest participant pool. Future work will
focus on expanding the range of tactile patterns and increasing user studies to
further refine and validate the system's performance. Overall, HapticVLM
presents a significant step toward context-aware, multimodal haptic interaction
with potential applications in virtual reality, and assistive technologies.

### 5. Riemannian Direct Trajectory Optimization of Rigid Bodies on Matrix Lie Groups

[Riemannian Direct Trajectory Optimization of Rigid Bodies on Matrix Lie Groups](http://arxiv.org/pdf/2505.02323v1)

Authors: Sangli Teng, Tzu-Yuan Lin, William A Clark, Ram Vasudevan, Maani Ghaffari

Designing dynamically feasible trajectories for rigid bodies is a fundamental
problem in robotics. Although direct trajectory optimization is widely applied
to solve this problem, inappropriate parameterizations of rigid body dynamics
often result in slow convergence and violations of the intrinsic topological
structure of the rotation group. This paper introduces a Riemannian
optimization framework for direct trajectory optimization of rigid bodies. We
first use the Lie Group Variational Integrator to formulate the discrete rigid
body dynamics on matrix Lie groups. We then derive the closed-form first- and
second-order Riemannian derivatives of the dynamics. Finally, this work applies
a line-search Riemannian Interior Point Method (RIPM) to perform trajectory
optimization with general nonlinear constraints. As the optimization is
performed on matrix Lie groups, it is correct-by-construction to respect the
topological structure of the rotation group and be free of singularities. The
paper demonstrates that both the derivative evaluations and Newton steps
required to solve the RIPM exhibit linear complexity with respect to the
planning horizon and system degrees of freedom. Simulation results illustrate
that the proposed method is faster than conventional methods by an order of
magnitude in challenging robotics tasks.

### 6. MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans

[MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans](http://arxiv.org/pdf/2505.02388v1)

Authors: Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang

Embodied AI (EAI) research requires high-quality, diverse 3D scenes to
effectively support skill acquisition, sim-to-real transfer, and
generalization. Achieving these quality standards, however, necessitates the
precise replication of real-world object diversity. Existing datasets
demonstrate that this process heavily relies on artist-driven designs, which
demand substantial human effort and present significant scalability challenges.
To scalably produce realistic and interactive 3D scenes, we first present
MetaScenes, a large-scale, simulatable 3D scene dataset constructed from
real-world scans, which includes 15366 objects spanning 831 fine-grained
categories. Then, we introduce Scan2Sim, a robust multi-modal alignment model,
which enables the automated, high-quality replacement of assets, thereby
eliminating the reliance on artist-driven designs for scaling 3D scenes. We
further propose two benchmarks to evaluate MetaScenes: a detailed scene
synthesis task focused on small item layouts for robotic manipulation and a
domain transfer task in vision-and-language navigation (VLN) to validate
cross-domain transfer. Results confirm MetaScene's potential to enhance EAI by
supporting more generalizable agent learning and sim-to-real applications,
introducing new possibilities for EAI research. Project website:
https://meta-scenes.github.io/.

### 7. A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints

[A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints](http://arxiv.org/pdf/2505.02395v1)

Authors: Jianye Xu, Chang Che, Bassam Alrifaee

We present a real-time safety filter for motion planning, such as
learning-based methods, using Control Barrier Functions (CBFs), which provides
formal guarantees for collision avoidance with road boundaries. A key feature
of our approach is its ability to directly incorporate road geometries of
arbitrary shape without resorting to conservative overapproximations. We
formulate the safety filter as a constrained optimization problem in the form
of a Quadratic Program (QP). It achieves safety by making minimal, necessary
adjustments to the control actions issued by the nominal motion planner. We
validate our safety filter through extensive numerical experiments across a
variety of traffic scenarios featuring complex roads. The results confirm its
reliable safety and high computational efficiency (execution frequency up to 40
Hz). Code & Video Demo: github.com/bassamlab/SigmaRL

### 8. Quadrupedal Spine Control Strategies: Exploring Correlations Between System Dynamic Responses and Human Perspectives

[Quadrupedal Spine Control Strategies: Exploring Correlations Between System Dynamic Responses and Human Perspectives](http://arxiv.org/pdf/2505.02414v1)

Authors: Nicholas Hafner, Chaoran Liu, Carlos Ishi, Hiroshi Ishiguro

Unlike their biological cousins, the majority of existing quadrupedal robots
are constructed with rigid chassis. This results in motion that is either
beetle-like or distinctly robotic, lacking the natural fluidity characteristic
of mammalian movements. Existing literature on quadrupedal robots with spinal
configurations primarily focuses on energy efficiency and does not consider the
effects in human-robot interaction scenarios. Our contributions include an
initial investigation into various trajectory generation strategies for a
quadrupedal robot with a four degree of freedom spine, and an analysis on the
effect that such methods have on human perception of gait naturalness compared
to a fixed spine baseline. The strategies were evaluated using videos of
walking, trotting and turning simulations. Among the four different strategies
developed, the optimised time varying and the foot-tracking strategies were
perceived to be more natural than the baseline in a randomised trial with 50
participants. Although none of the strategies demonstrated any energy
efficiency improvements over the no-spine baseline, some showed greater
footfall consistency at higher speeds. Given the greater likeability drawn from
the more natural locomotion patterns, this type of robot displays potential for
applications in social robot scenarios such as elderly care, where energy
efficiency is not a primary concern.

### 9. ZeloS -- A Research Platform for Early-Stage Validation of Research Findings Related to Automated Driving

[ZeloS -- A Research Platform for Early-Stage Validation of Research Findings Related to Automated Driving](http://arxiv.org/pdf/2505.02460v1)

Authors: Christopher Bohn, Florian Siebenrock, Janne Bosch, Tobias Hetzner, Samuel Mauch, Philipp Reis, Timo Staudt, Manuel Hess, Ben-Micha Piscol, Sören Hohmann

This paper presents ZeloS, a research platform designed and built for
practical validation of automated driving methods in an early stage of
research. We overview ZeloS' hardware setup and automation architecture and
focus on motion planning and control. ZeloS weighs 69 kg, measures a length of
117 cm, and is equipped with all-wheel steering, all-wheel drive, and various
onboard sensors for localization. The hardware setup and the automation
architecture of ZeloS are designed and built with a focus on modularity and the
goal of being simple yet effective. The modular design allows the modification
of individual automation modules without the need for extensive onboarding into
the automation architecture. As such, this design supports ZeloS in being a
versatile research platform for validating various automated driving methods.
The motion planning component and control of ZeloS feature optimization-based
methods that allow for explicitly considering constraints. We demonstrate the
hardware and automation setup by presenting experimental data.

### 10. Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation

[Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation](http://arxiv.org/pdf/2505.02476v1)

Authors: Hubert Padusinski, Christian Steinhauser, Christian Scherl, Julian Gaal, Jacob Langner

The validation of LiDAR-based perception of intelligent mobile systems
operating in open-world applications remains a challenge due to the variability
of real environmental conditions. Virtual simulations allow the generation of
arbitrary scenes under controlled conditions but lack physical sensor
characteristics, such as intensity responses or material-dependent effects. In
contrast, real-world data offers true sensor realism but provides less control
over influencing factors, hindering sufficient validation. Existing approaches
address this problem with augmentation of real-world point cloud data by
transferring objects between scenes. However, these methods do not consider
validation and remain limited in controllability because they rely on empirical
data. We solve these limitations by proposing Point Cloud Recombination, which
systematically augments captured point cloud scenes by integrating point clouds
acquired from physical target objects measured in controlled laboratory
environments. Thus enabling the creation of vast amounts and varieties of
repeatable, physically accurate test scenes with respect to phenomena-aware
occlusions with registered 3D meshes. Using the Ouster OS1-128 Rev7 sensor, we
demonstrate the augmentation of real-world urban and rural scenes with humanoid
targets featuring varied clothing and poses, for repeatable positioning. We
show that the recombined scenes closely match real sensor outputs, enabling
targeted testing, scalable failure analysis, and improved system safety. By
providing controlled yet sensor-realistic data, our method enables trustworthy
conclusions about the limitations of specific sensors in compound with their
algorithms, e.g., object detection.

### Software Engineering

### 1. Refining Fuzzed Crashing Inputs for Better Fault Diagnosis

[Refining Fuzzed Crashing Inputs for Better Fault Diagnosis](http://arxiv.org/pdf/2505.02305v1)

Authors: Kieun Kim, Seongmin Lee, Shin Hong

We present DiffMin, a technique that refines a fuzzed crashing input to gain
greater similarities to given passing inputs to help developers analyze the
crashing input to identify the failure-inducing condition and locate buggy code
for debugging. DiffMin iteratively applies edit actions to transform a fuzzed
input while preserving the crash behavior. Our pilot study with the Magma
benchmark demonstrates that DiffMin effectively minimizes the differences
between crashing and passing inputs while enhancing the accuracy of
spectrum-based fault localization, highlighting its potential as a valuable
pre-debugging step after greybox fuzzing.

### 2. RouthSearch: Inferring PID Parameter Specification for Flight Control Program by Coordinate Search

[RouthSearch: Inferring PID Parameter Specification for Flight Control Program by Coordinate Search](http://arxiv.org/pdf/2505.02357v1)

Authors: Siao Wang, Zhen Dong, Hui Li, Liwei Shen, Xin Peng, Dongdong She

Flight control programs use PID control modules with user-configurable
Proportional (P), Integral (I), and Derivative (D) parameters to manage UAV
flying behaviors. Users can adjust these PID parameters during flight. However,
flight control programs lack sufficient safety checks on user-provided PID
parameters, leading to a severe UAV vulnerability - the input validation bug.
This occurs when a user misconfigures PID parameters, causing dangerous states
like deviation from the expected path, loss of control, or crash.
  Prior works use random testing like fuzzing, but these are not effective in
the three-dimensional search space of PID parameters. The expensive dynamic
execution of UAV tests further hinders random testing performance.
  We address PID parameter misconfiguration by combining the Routh-Hurwitz
stability criterion with coordinate search, introducing RouthSearch. Instead of
ad-hoc identification, RouthSearch principledly determines valid ranges for
three-dimensional PID parameters. We first leverage the Routh-Hurwitz Criterion
to identify a theoretical PID parameter boundary, then refine it using
efficient coordinate search. The determined valid range can filter
misconfigured PID parameters from users during flight and help discover logical
bugs in flight control programs.
  We evaluated RouthSearch across eight flight modes in PX4 and Ardupilot.
Results show RouthSearch determines valid ranges with 92.0% accuracy compared
to ground truth. RouthSearch discovers 3,853 PID misconfigurations within 48
hours, while the STOA work PGFuzz discovers only 449 sets, significantly
outperforming prior works by 8.58 times. Our method also helped detect three
bugs in ArduPilot and PX4.

### 3. LAMeD: LLM-generated Annotations for Memory Leak Detection

[LAMeD: LLM-generated Annotations for Memory Leak Detection](http://arxiv.org/pdf/2505.02376v1)

Authors: Ekaterina Shemetova, Ilya Shenbin, Ivan Smirnov, Anton Alekseev, Alexey Rukhovich, Sergey Nikolenko, Vadim Lomshakov, Irina Piontkovskaya

Static analysis tools are widely used to detect software bugs and
vulnerabilities but often struggle with scalability and efficiency in complex
codebases. Traditional approaches rely on manually crafted annotations --
labeling functions as sources or sinks -- to track data flows, e.g., ensuring
that allocated memory is eventually freed, and code analysis tools such as
CodeQL, Infer, or Cooddy can use function specifications, but manual annotation
is laborious and error-prone, especially for large or third-party libraries. We
present LAMeD (LLM-generated Annotations for Memory leak Detection), a novel
approach that leverages large language models (LLMs) to automatically generate
function-specific annotations. When integrated with analyzers such as Cooddy,
LAMeD significantly improves memory leak detection and reduces path explosion.
We also suggest directions for extending LAMeD to broader code analysis.

### 4. Towards Effective Issue Assignment using Online Machine Learning

[Towards Effective Issue Assignment using Online Machine Learning](http://arxiv.org/pdf/2505.02437v1)

Authors: Athanasios Michailoudis, Themistoklis Diamantopoulos, Antonios Favvas, Andreas L. Symeonidis

Efficient issue assignment in software development relates to faster
resolution time, resources optimization, and reduced development effort. To
this end, numerous systems have been developed to automate issue assignment,
including AI and machine learning approaches. Most of them, however, often
solely focus on a posteriori analyses of textual features (e.g. issue titles,
descriptions), disregarding the temporal characteristics of software
development. Thus, they fail to adapt as projects and teams evolve, such cases
of team evolution, or project phase shifts (e.g. from development to
maintenance). To incorporate such cases in the issue assignment process, we
propose an Online Machine Learning methodology that adapts to the evolving
characteristics of software projects. Our system processes issues as a data
stream, dynamically learning from new data and adjusting in real time to
changes in team composition and project requirements. We incorporate metadata
such as issue descriptions, components and labels and leverage adaptive drift
detection mechanisms to identify when model re-evaluation is necessary. Upon
assessing our methodology on a set of software projects, we conclude that it
can be effective on issue assignment, while meeting the evolving needs of
software teams.

### 5. Automating Automotive Software Development: A Synergy of Generative AI and Formal Methods

[Automating Automotive Software Development: A Synergy of Generative AI and Formal Methods](http://arxiv.org/pdf/2505.02500v1)

Authors: Fengjunjie Pan, Yinglei Song, Long Wen, Nenad Petrovic, Krzysztof Lebioda, Alois Knoll

As the automotive industry shifts its focus toward software-defined vehicles,
the need for faster and reliable software development continues to grow.
However, traditional methods show their limitations. The rise of Generative
Artificial Intelligence (GenAI), particularly Large Language Models (LLMs),
introduces new opportunities to automate automotive software development tasks
such as requirement analysis and code generation. However, due to the
complexity of automotive systems, where software components must interact with
each other seamlessly, challenges remain in software integration and
system-level validation. In this paper, we propose to combine GenAI with
model-driven engineering to automate automotive software development. Our
approach uses LLMs to convert free-text requirements into event chain
descriptions and to generate platform-independent software components that
realize the required functionality. At the same time, formal models are created
based on event chain descriptions to support system validation and the
generation of integration code for integrating generated software components in
the whole vehicle system through middleware. This approach increases
development automation while enabling formal analysis to improve system
reliability. As a proof of concept, we used GPT-4o to implement our method and
tested it in the CARLA simulation environment with ROS2 middleware. We
evaluated the system in a simple Autonomous Emergency Braking scenario.

### 6. Parameter-Efficient Fine-Tuning with Attributed Patch Semantic Graph for Automated Patch Correctness Assessment

[Parameter-Efficient Fine-Tuning with Attributed Patch Semantic Graph for Automated Patch Correctness Assessment](http://arxiv.org/pdf/2505.02629v1)

Authors: Zhenyu Yang, Jingwen Wu, Zhen Yang, Zhongxing Yu

Automated program repair (APR) aims to automatically repair program errors
without human intervention, and recent years have witnessed a growing interest
on this research topic. While much progress has been made and techniques
originating from different disciplines have been proposed, APR techniques
generally suffer from the patch overfitting issue, i.e., the generated patches
are not genuinely correct despite they pass the employed tests. To alleviate
this issue, many research efforts have been devoted for automated patch
correctness assessment (APCA). In particular, with the emergence of large
language model (LLM) technology, researchers have employed LLM to assess the
patch correctness and have obtained the state-of-the-art performance. The
literature on APCA has demonstrated the importance of capturing patch semantic
and explicitly considering certain code attributes in predicting patch
correctness. However, existing LLM-based methods typically treat code as token
sequences and ignore the inherent formal structure for code, making it
difficult to capture the deep patch semantics. Moreover, these LLM-based
methods also do not explicitly account for enough code attributes. To overcome
these drawbacks, we in this paper design a novel patch graph representation
named attributed patch semantic graph (APSG), which adequately captures the
patch semantic and explicitly reflects important patch attributes. To
effectively use graph information in APSG, we accordingly propose a new
parameter-efficient fine-tuning (PEFT) method of LLMs named Graph-LoRA.
Extensive evaluations have been conducted to evaluate our method, and the
results show that compared to the state-of-the-art methods, our method improves
accuracy and F1 score by 2.3% to 6.6% and 1.8% to 6.1% respectively.

### 7. A Slicing-Based Approach for Detecting and Patching Vulnerable Code Clones

[A Slicing-Based Approach for Detecting and Patching Vulnerable Code Clones](http://arxiv.org/pdf/2505.02349v1)

Authors: Hakam Alomari, Christopher Vendome, Hilal Gyawali

Code cloning is a common practice in software development, but it poses
significant security risks by propagating vulnerabilities across cloned
segments. To address this challenge, we introduce srcVul, a scalable, precise
detection approach that combines program slicing with Locality-Sensitive
Hashing to identify vulnerable code clones and recommend patches. srcVul builds
a database of vulnerability-related slices by analyzing known vulnerable
programs and their corresponding patches, indexing each slice's unique
structural characteristics as a vulnerability slicing vector. During clone
detection, srcVul efficiently matches slicing vectors from target programs with
those in the database, recommending patches upon identifying similarities. Our
evaluation of srcVul against three state-of-the-art vulnerable clone detectors
demonstrates its accuracy, efficiency, and scalability, achieving 91% precision
and 75% recall on established vulnerability databases and open-source
repositories. These results highlight srcVul's effectiveness in detecting
complex vulnerability patterns across diverse codebases.

### 8. A Unifying Framework to Enable Artificial Intelligence in High Performance Computing Workflows

[A Unifying Framework to Enable Artificial Intelligence in High Performance Computing Workflows](http://arxiv.org/pdf/2505.02738v1)

Authors: Jens Domke, Mohamed Wahib, Anshu Dubey, Tal Ben-Nun, Erik W. Draeger

Current trends point to a future where large-scale scientific applications
are tightly-coupled HPC/AI hybrids. Hence, we urgently need to invest in
creating a seamless, scalable framework where HPC and AI/ML can efficiently
work together and adapt to novel hardware and vendor libraries without starting
from scratch every few years. The current ecosystem and sparsely-connected
community are not sufficient to tackle these challenges, and we require a
breakthrough catalyst for science similar to what PyTorch enabled for AI.

### 9. Regulating Algorithmic Management: A Multi-Stakeholder Study of Challenges in Aligning Software and the Law for Workplace Scheduling

[Regulating Algorithmic Management: A Multi-Stakeholder Study of Challenges in Aligning Software and the Law for Workplace Scheduling](http://arxiv.org/pdf/2505.02329v1)

Authors: Jonathan Lynn, Rachel Y. Kim, Sicun Gao, Daniel Schneider, Sachin S. Pandya, Min Kyung Lee

The impacts of algorithmic management (AM) on worker well-being have led to
increasing calls to regulate AM practices to prevent further worker harms. Yet
existing work in aligning software with the law reduces compliance to just one
piece of the entire process of regulating AM -- which involves rule
operationalization, software use, and enforcement. We interviewed key
stakeholders involved in enforcing or complying with workplace scheduling law
-- regulators, advocates, defense attorneys, scheduling managers, and workers
($N = 38$). Based on their beliefs and experiences, we describe how scheduling
software affects beliefs about and compliance with workplace scheduling law. In
so doing, we discuss the challenges and opportunities in designing software as
a tool for regulating AM.

### 10. An Empirical Study on the Performance and Energy Usage of Compiled Python Code

[An Empirical Study on the Performance and Energy Usage of Compiled Python Code](http://arxiv.org/pdf/2505.02346v1)

Authors: Vincenzo Stoico, Andrei Calin Dragomir, Patricia Lago

Python is a popular programming language known for its ease of learning and
extensive libraries. However, concerns about performance and energy consumption
have led to the development of compilers to enhance Python code efficiency.
Despite the proven benefits of existing compilers on the efficiency of Python
code, there is limited analysis comparing their performance and energy
efficiency, particularly considering code characteristics and factors like CPU
frequency and core count. Our study investigates how compilation impacts the
performance and energy consumption of Python code, using seven benchmarks
compiled with eight different tools: PyPy, Numba, Nuitka, Mypyc, Codon, Cython,
Pyston-lite, and the experimental Python 3.13 version, compared to CPython. The
benchmarks are single-threaded and executed on an NUC and a server, measuring
energy usage, execution time, memory usage, and Last-Level Cache (LLC) miss
rates at a fixed frequency and on a single core. The results show that
compilation can significantly enhance execution time, energy and memory usage,
with Codon, PyPy, and Numba achieving over 90\% speed and energy improvements.
Nuitka optimizes memory usage consistently on both testbeds. The impact of
compilation on LLC miss rate is not clear since it varies considerably across
benchmarks for each compiler. Our study is important for researchers and
practitioners focused on improving Python code performance and energy
efficiency. We outline future research directions, such as exploring caching
effects on energy usage. Our findings help practitioners choose the best
compiler based on their efficiency benefits and accessibility.

### Social and Information Networks

### 1. A longitudinal analysis of misinformation, polarization and toxicity on Bluesky after its public launch

[A longitudinal analysis of misinformation, polarization and toxicity on Bluesky after its public launch](http://arxiv.org/pdf/2505.02317v1)

Authors: Gianluca Nogara, Erfan Samieyan Sahneh, Matthew R. DeVerna, Nick Liu, Luca Luceri, Filippo Menczer, Francesco Pierri, Silvia Giordano

Bluesky is a decentralized, Twitter-like social media platform that has
rapidly gained popularity. Following an invite-only phase, it officially opened
to the public on February 6th, 2024, leading to a significant expansion of its
user base. In this paper, we present a longitudinal analysis of user activity
in the two months surrounding its public launch, examining how the platform
evolved due to this rapid growth. Our analysis reveals that Bluesky exhibits an
activity distribution comparable to more established social platforms, yet it
features a higher volume of original content relative to reshared posts and
maintains low toxicity levels. We further investigate the political leanings of
its user base, misinformation dynamics, and engagement in harmful
conversations. Our findings indicate that Bluesky users predominantly lean left
politically and tend to share high-credibility sources. After the platform's
public launch, an influx of new users, particularly those posting in English
and Japanese, contributed to a surge in activity. Among them, several accounts
displayed suspicious behaviors, such as mass-following users and sharing
content from low-credibility news sources. Some of these accounts have already
been flagged as spam or suspended, suggesting that Bluesky's moderation efforts
have been effective.

### 2. Social Correction on Social Media: A Quantitative Analysis of Comment Behaviour and Reliability

[Social Correction on Social Media: A Quantitative Analysis of Comment Behaviour and Reliability](http://arxiv.org/pdf/2505.02343v1)

Authors: Sameera S. Vithanage, Keith Ransom, Antonette Mendoza, Shanika Karunasekera

Corrections given by ordinary social media users, also referred to as Social
Correction have emerged as a viable intervention against misinformation as per
the recent literature. However, little is known about how often users give
disputing or endorsing comments and how reliable those comments are. An online
experiment was conducted to investigate how users' credibility evaluations of
social media posts and their confidence in those evaluations combined with
online reputational concerns affect their commenting behaviour. The study found
that participants exhibited a more conservative approach when giving disputing
comments compared to endorsing ones. Nevertheless, participants were more
discerning in their disputing comments than endorsing ones. These findings
contribute to a better understanding of social correction on social media and
highlight the factors influencing comment behaviour and reliability.

### 3. Rethinking Federated Graph Learning: A Data Condensation Perspective

[Rethinking Federated Graph Learning: A Data Condensation Perspective](http://arxiv.org/pdf/2505.02573v1)

Authors: Hao Zhang, Xunkai Li, Yinlin Zhu, Lianglin Hu

Federated graph learning is a widely recognized technique that promotes
collaborative training of graph neural networks (GNNs) by multi-client
graphs.However, existing approaches heavily rely on the communication of model
parameters or gradients for federated optimization and fail to adequately
address the data heterogeneity introduced by intricate and diverse graph
distributions. Although some methods attempt to share additional messages among
the server and clients to improve federated convergence during communication,
they introduce significant privacy risks and increase communication overhead.
To address these issues, we introduce the concept of a condensed graph as a
novel optimization carrier to address FGL data heterogeneity and propose a new
FGL paradigm called FedGM. Specifically, we utilize a generalized condensation
graph consensus to aggregate comprehensive knowledge from distributed graphs,
while minimizing communication costs and privacy risks through a single
transmission of the condensed data. Extensive experiments on six public
datasets consistently demonstrate the superiority of FedGM over
state-of-the-art baselines, highlighting its potential for a novel FGL
paradigm.

### 4. dyGRASS: Dynamic Spectral Graph Sparsification via Localized Random Walks on GPUs

[dyGRASS: Dynamic Spectral Graph Sparsification via Localized Random Walks on GPUs](http://arxiv.org/pdf/2505.02741v1)

Authors: Yihang Yuan, Ali Aghdaei, Zhuo Feng

This work presents dyGRASS, an efficient dynamic algorithm for spectral
sparsification of large undirected graphs that undergo streaming edge
insertions and deletions. At its core, dyGRASS employs a random-walk-based
method to efficiently estimate node-to-node distances in both the original
graph (for decremental update) and its sparsifier (for incremental update). For
incremental updates, dyGRASS enables the identification of spectrally critical
edges among the updates to capture the latest structural changes. For
decremental updates, dyGRASS facilitates the recovery of important edges from
the original graph back into the sparsifier. To further enhance computational
efficiency, dyGRASS employs a GPU-based non-backtracking random walk scheme
that allows multiple walkers to operate simultaneously across various target
updates. This parallelization significantly improves both the performance and
scalability of the proposed dyGRASS framework. Our comprehensive experimental
evaluations reveal that dyGRASS achieves approximately a 10x speedup compared
to the state-of-the-art incremental sparsification (inGRASS) algorithm while
eliminating the setup overhead and improving solution quality in incremental
spectral sparsification tasks. Moreover, dyGRASS delivers high efficiency and
superior solution quality for fully dynamic graph sparsification, accommodating
both edge insertions and deletions across a diverse range of graph instances
originating from integrated circuit simulations, finite element analysis, and
social networks.

### Systems and Control

### 1. Impact of Transceiver Selection on Synchronization Accuracy in White Rabbit Networks

[Impact of Transceiver Selection on Synchronization Accuracy in White Rabbit Networks](http://arxiv.org/pdf/2505.02420v1)

Authors: Michal Špaček, Josef Vojtěch, Jaroslav Roztočil

Achieving optimal synchronization accuracy between two White Rabbit devices
hinges on the proper selection of transceivers, which act as electro-optical
converters connecting WR devices to the optical network infrastructure. The
correct choice of transceivers can significantly improve resilience to changes
in the time offset between WR devices due to temperature fluctuations in the
connecting optical fiber. To compare the performance of BiDi WDM and DWDM
transceivers, an experimental setup was established under laboratory conditions
to simulate a real optical network used for distributing precise time and
frequency between two remote locations. The optical connection was emulated by
integrating a 20 km G.652.D optical fiber into a climatic chamber, which
provided variable environmental conditions similar to those experienced in real
applications. The study compared BiDi WDM 1310/1550 nm transceivers with DWDM
Ch33/Ch34 transceivers. Results showed that DWDM transceivers exhibited nearly
thirteen times less sensitivity to temperature-induced changes in the optical
connection, leading to a smaller time offset. Therefore, for achieving the
highest accuracy in synchronizing WR devices in practical applications, DWDM
transceiver technology is essential.

### 2. Maximal Compatibility Matching for Preference-Aware Ride-Hailing Systems

[Maximal Compatibility Matching for Preference-Aware Ride-Hailing Systems](http://arxiv.org/pdf/2505.02599v1)

Authors: Avalpreet Singh Brar, Rong Su, Jaskaranveer Kaur, Xinling Li, Gioele Zardini

This paper presents the Maximal Compatibility Matching (MCM) framework, a
novel assignment strategy for ride-hailing systems that explicitly incorporates
passenger comfort into the matching process. Traditional assignment methods
prioritize spatial efficiency, but often overlook behavioral alignment between
passengers and drivers, which can significantly impact user satisfaction. MCM
addresses this gap by learning personalized passenger comfort zones using
gradient-boosted decision tree classifiers trained on labeled ride data, and by
modeling driver behavior through empirical operating profiles constructed from
time-series driving features. Compatibility between a passenger and a driver is
computed as the closed-form volume of intersection between their respective
feature-space regions. These compatibility scores are integrated into a
utility-based matching algorithm that balances comfort and proximity through a
tunable trade-off parameter. We validate the framework using a Unity-based
driving simulator with real-time passenger feedback, demonstrating that MCM
enables more personalized and socially acceptable matchings while maintaining
high levels of operational performance.

### 3. Wise Goose Chase: A Predictive Path Planning Algorithm for Dynamic Rebalancing in Ride-Hailing Systems

[Wise Goose Chase: A Predictive Path Planning Algorithm for Dynamic Rebalancing in Ride-Hailing Systems](http://arxiv.org/pdf/2505.02603v1)

Authors: Avalpreet Singh Brar, Rong Su, Christos G. Cassandras, Gioele Zardini

Traditional rebalancing methods in ride-hailing systems direct idle drivers
to fixed destinations, overlooking the fact that ride allocations frequently
occur while cruising. This destination-centric view fails to exploit the
path-dependent nature of modern platforms, where real-time matching depends on
the entire trajectory rather than a static endpoint. We propose the Wise Goose
Chase (WGC) algorithm, an event-triggered, driver-specific path planning
framework that anticipates future matching opportunities by forecasting
spatio-temporal supply and demand dynamics. WGC uses a system of Retarded
Functional Differential Equations (RFDEs) to model the evolution of idle driver
density and passenger queues at the road-segment level, incorporating both
en-route matching and competition among drivers. Upon request, WGC computes
personalized cruising paths that minimize each driver's expected time to
allocation. Monte Carlo simulations on synthetic urban networks show that WGC
consistently outperforms baseline strategies, highlighting the advantage of
predictive, context-aware rebalancing in dynamic mobility systems.

### 4. Online Phase Estimation of Human Oscillatory Motions using Deep Learning

[Online Phase Estimation of Human Oscillatory Motions using Deep Learning](http://arxiv.org/pdf/2505.02668v1)

Authors: Antonio Grotta, Francesco De Lellis

Accurately estimating the phase of oscillatory systems is essential for
analyzing cyclic activities such as repetitive gestures in human motion. In
this work we introduce a learning-based approach for online phase estimation in
three-dimensional motion trajectories, using a Long Short- Term Memory (LSTM)
network. A calibration procedure is applied to standardize trajectory position
and orientation, ensuring invariance to spatial variations. The proposed model
is evaluated on motion capture data and further tested in a dynamical system,
where the estimated phase is used as input to a reinforcement learning
(RL)-based control to assess its impact on the synchronization of a network of
Kuramoto oscillators.

### 5. Riemannian Direct Trajectory Optimization of Rigid Bodies on Matrix Lie Groups

[Riemannian Direct Trajectory Optimization of Rigid Bodies on Matrix Lie Groups](http://arxiv.org/pdf/2505.02323v1)

Authors: Sangli Teng, Tzu-Yuan Lin, William A Clark, Ram Vasudevan, Maani Ghaffari

Designing dynamically feasible trajectories for rigid bodies is a fundamental
problem in robotics. Although direct trajectory optimization is widely applied
to solve this problem, inappropriate parameterizations of rigid body dynamics
often result in slow convergence and violations of the intrinsic topological
structure of the rotation group. This paper introduces a Riemannian
optimization framework for direct trajectory optimization of rigid bodies. We
first use the Lie Group Variational Integrator to formulate the discrete rigid
body dynamics on matrix Lie groups. We then derive the closed-form first- and
second-order Riemannian derivatives of the dynamics. Finally, this work applies
a line-search Riemannian Interior Point Method (RIPM) to perform trajectory
optimization with general nonlinear constraints. As the optimization is
performed on matrix Lie groups, it is correct-by-construction to respect the
topological structure of the rotation group and be free of singularities. The
paper demonstrates that both the derivative evaluations and Newton steps
required to solve the RIPM exhibit linear complexity with respect to the
planning horizon and system degrees of freedom. Simulation results illustrate
that the proposed method is faster than conventional methods by an order of
magnitude in challenging robotics tasks.

### 6. A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints

[A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints](http://arxiv.org/pdf/2505.02395v1)

Authors: Jianye Xu, Chang Che, Bassam Alrifaee

We present a real-time safety filter for motion planning, such as
learning-based methods, using Control Barrier Functions (CBFs), which provides
formal guarantees for collision avoidance with road boundaries. A key feature
of our approach is its ability to directly incorporate road geometries of
arbitrary shape without resorting to conservative overapproximations. We
formulate the safety filter as a constrained optimization problem in the form
of a Quadratic Program (QP). It achieves safety by making minimal, necessary
adjustments to the control actions issued by the nominal motion planner. We
validate our safety filter through extensive numerical experiments across a
variety of traffic scenarios featuring complex roads. The results confirm its
reliable safety and high computational efficiency (execution frequency up to 40
Hz). Code & Video Demo: github.com/bassamlab/SigmaRL

### 7. Quadrupedal Spine Control Strategies: Exploring Correlations Between System Dynamic Responses and Human Perspectives

[Quadrupedal Spine Control Strategies: Exploring Correlations Between System Dynamic Responses and Human Perspectives](http://arxiv.org/pdf/2505.02414v1)

Authors: Nicholas Hafner, Chaoran Liu, Carlos Ishi, Hiroshi Ishiguro

Unlike their biological cousins, the majority of existing quadrupedal robots
are constructed with rigid chassis. This results in motion that is either
beetle-like or distinctly robotic, lacking the natural fluidity characteristic
of mammalian movements. Existing literature on quadrupedal robots with spinal
configurations primarily focuses on energy efficiency and does not consider the
effects in human-robot interaction scenarios. Our contributions include an
initial investigation into various trajectory generation strategies for a
quadrupedal robot with a four degree of freedom spine, and an analysis on the
effect that such methods have on human perception of gait naturalness compared
to a fixed spine baseline. The strategies were evaluated using videos of
walking, trotting and turning simulations. Among the four different strategies
developed, the optimised time varying and the foot-tracking strategies were
perceived to be more natural than the baseline in a randomised trial with 50
participants. Although none of the strategies demonstrated any energy
efficiency improvements over the no-spine baseline, some showed greater
footfall consistency at higher speeds. Given the greater likeability drawn from
the more natural locomotion patterns, this type of robot displays potential for
applications in social robot scenarios such as elderly care, where energy
efficiency is not a primary concern.

### 8. ReeM: Ensemble Building Thermodynamics Model for Efficient HVAC Control via Hierarchical Reinforcement Learning

[ReeM: Ensemble Building Thermodynamics Model for Efficient HVAC Control via Hierarchical Reinforcement Learning](http://arxiv.org/pdf/2505.02439v1)

Authors: Yang Deng, Yaohui Liu, Rui Liang, Dafang Zhao, Donghua Xie, Ittetsu Taniguchi, Dan Wang

The building thermodynamics model, which predicts real-time indoor
temperature changes under potential HVAC (Heating, Ventilation, and Air
Conditioning) control operations, is crucial for optimizing HVAC control in
buildings. While pioneering studies have attempted to develop such models for
various building environments, these models often require extensive data
collection periods and rely heavily on expert knowledge, making the modeling
process inefficient and limiting the reusability of the models. This paper
explores a model ensemble perspective that utilizes existing developed models
as base models to serve a target building environment, thereby providing
accurate predictions while reducing the associated efforts. Given that building
data streams are non-stationary and the number of base models may increase, we
propose a Hierarchical Reinforcement Learning (HRL) approach to dynamically
select and weight the base models. Our approach employs a two-tiered
decision-making process: the high-level focuses on model selection, while the
low-level determines the weights of the selected models. We thoroughly evaluate
the proposed approach through offline experiments and an on-site case study,
and the experimental results demonstrate the effectiveness of our method.

### 9. ZeloS -- A Research Platform for Early-Stage Validation of Research Findings Related to Automated Driving

[ZeloS -- A Research Platform for Early-Stage Validation of Research Findings Related to Automated Driving](http://arxiv.org/pdf/2505.02460v1)

Authors: Christopher Bohn, Florian Siebenrock, Janne Bosch, Tobias Hetzner, Samuel Mauch, Philipp Reis, Timo Staudt, Manuel Hess, Ben-Micha Piscol, Sören Hohmann

This paper presents ZeloS, a research platform designed and built for
practical validation of automated driving methods in an early stage of
research. We overview ZeloS' hardware setup and automation architecture and
focus on motion planning and control. ZeloS weighs 69 kg, measures a length of
117 cm, and is equipped with all-wheel steering, all-wheel drive, and various
onboard sensors for localization. The hardware setup and the automation
architecture of ZeloS are designed and built with a focus on modularity and the
goal of being simple yet effective. The modular design allows the modification
of individual automation modules without the need for extensive onboarding into
the automation architecture. As such, this design supports ZeloS in being a
versatile research platform for validating various automated driving methods.
The motion planning component and control of ZeloS feature optimization-based
methods that allow for explicitly considering constraints. We demonstrate the
hardware and automation setup by presenting experimental data.

### 10. Data-Driven Energy Modeling of Industrial IoT Systems: A Benchmarking Approach

[Data-Driven Energy Modeling of Industrial IoT Systems: A Benchmarking Approach](http://arxiv.org/pdf/2505.02543v1)

Authors: Dimitris Kallis, Moysis Symeonides, Marios D. Dikaiakos

The widespread adoption of IoT has driven the development of cyber-physical
systems (CPS) in industrial environments, leveraging Industrial IoTs (IIoTs) to
automate manufacturing processes and enhance productivity. The transition to
autonomous systems introduces significant operational costs, particularly in
terms of energy consumption. Accurate modeling and prediction of IIoT energy
requirements are critical, but traditional physics- and engineering-based
approaches often fall short in addressing these challenges comprehensively. In
this paper, we propose a novel methodology for benchmarking and analyzing IIoT
devices and applications to uncover insights into their power demands, energy
consumption, and performance. To demonstrate this methodology, we develop a
comprehensive framework and apply it to study an industrial CPS comprising an
educational robotic arm, a conveyor belt, a smart camera, and a compute node.
By creating micro-benchmarks and an end-to-end application within this
framework, we create an extensive performance and power consumption dataset,
which we use to train and analyze ML models for predicting energy usage from
features of the application and the CPS system. The proposed methodology and
framework provide valuable insights into the energy dynamics of industrial CPS,
offering practical implications for researchers and practitioners aiming to
enhance the efficiency and sustainability of IIoT-driven automation.

### Machine Learning (Statistics Category)

### 1. Entropy-Guided Sampling of Flat Modes in Discrete Spaces

[Entropy-Guided Sampling of Flat Modes in Discrete Spaces](http://arxiv.org/pdf/2505.02296v1)

Authors: Pinaki Mohanty, Riddhiman Bhattacharya, Ruqi Zhang

Sampling from flat modes in discrete spaces is a crucial yet underexplored
problem. Flat modes represent robust solutions and have broad applications in
combinatorial optimization and discrete generative modeling. However, existing
sampling algorithms often overlook the mode volume and struggle to capture flat
modes effectively. To address this limitation, we propose \emph{Entropic
Discrete Langevin Proposal} (EDLP), which incorporates local entropy into the
sampling process through a continuous auxiliary variable under a joint
distribution. The local entropy term guides the discrete sampler toward flat
modes with a small overhead. We provide non-asymptotic convergence guarantees
for EDLP in locally log-concave discrete distributions. Empirically, our method
consistently outperforms traditional approaches across tasks that require
sampling from flat basins, including Bernoulli distribution, restricted
Boltzmann machines, combinatorial optimization, and binary neural networks.

### 2. Bayesian Robust Aggregation for Federated Learning

[Bayesian Robust Aggregation for Federated Learning](http://arxiv.org/pdf/2505.02490v1)

Authors: Aleksandr Karakulev, Usama Zafar, Salman Toor, Prashant Singh

Federated Learning enables collaborative training of machine learning models
on decentralized data. This scheme, however, is vulnerable to adversarial
attacks, when some of the clients submit corrupted model updates. In real-world
scenarios, the total number of compromised clients is typically unknown, with
the extent of attacks potentially varying over time. To address these
challenges, we propose an adaptive approach for robust aggregation of model
updates based on Bayesian inference. The mean update is defined by the maximum
of the likelihood marginalized over probabilities of each client to be
`honest'. As a result, the method shares the simplicity of the classical
average estimators (e.g., sample mean or geometric median), being independent
of the number of compromised clients. At the same time, it is as effective
against attacks as methods specifically tailored to Federated Learning, such as
Krum. We compare our approach with other aggregation schemes in federated
setting on three benchmark image classification data sets. The proposed method
consistently achieves state-of-the-art performance across various attack types
with static and varying number of malicious clients.

### 3. Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era

[Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era](http://arxiv.org/pdf/2505.02583v1)

Authors: Chenxi Liu, Shaowen Zhou, Qianxiong Xu, Hao Miao, Cheng Long, Ziyue Li, Rui Zhao

The proliferation of edge devices has generated an unprecedented volume of
time series data across different domains, motivating various well-customized
methods. Recently, Large Language Models (LLMs) have emerged as a new paradigm
for time series analytics by leveraging the shared sequential nature of textual
data and time series. However, a fundamental cross-modality gap between time
series and LLMs exists, as LLMs are pre-trained on textual corpora and are not
inherently optimized for time series. Many recent proposals are designed to
address this issue. In this survey, we provide an up-to-date overview of
LLMs-based cross-modality modeling for time series analytics. We first
introduce a taxonomy that classifies existing approaches into four groups based
on the type of textual data employed for time series modeling. We then
summarize key cross-modality strategies, e.g., alignment and fusion, and
discuss their applications across a range of downstream tasks. Furthermore, we
conduct experiments on multimodal datasets from different application domains
to investigate effective combinations of textual data and cross-modality
strategies for enhancing time series analytics. Finally, we suggest several
promising directions for future research. This survey is designed for a range
of professionals, researchers, and practitioners interested in LLM-based time
series modeling.

### 4. Cooperative Bayesian and variance networks disentangle aleatoric and epistemic uncertainties

[Cooperative Bayesian and variance networks disentangle aleatoric and epistemic uncertainties](http://arxiv.org/pdf/2505.02743v1)

Authors: Jiaxiang Yi, Miguel A. Bessa

Real-world data contains aleatoric uncertainty - irreducible noise arising
from imperfect measurements or from incomplete knowledge about the data
generation process. Mean variance estimation (MVE) networks can learn this type
of uncertainty but require ad-hoc regularization strategies to avoid
overfitting and are unable to predict epistemic uncertainty (model
uncertainty). Conversely, Bayesian neural networks predict epistemic
uncertainty but are notoriously difficult to train due to the approximate
nature of Bayesian inference. We propose to cooperatively train a variance
network with a Bayesian neural network and demonstrate that the resulting model
disentangles aleatoric and epistemic uncertainties while improving the mean
estimation. We demonstrate the effectiveness and scalability of this method
across a diverse range of datasets, including a time-dependent heteroscedastic
regression dataset we created where the aleatoric uncertainty is known. The
proposed method is straightforward to implement, robust, and adaptable to
various model architectures.

### 5. A probabilistic view on Riemannian machine learning models for SPD matrices

[A probabilistic view on Riemannian machine learning models for SPD matrices](http://arxiv.org/pdf/2505.02402v1)

Authors: Thibault de Surrel, Florian Yger, Fabien Lotte, Sylvain Chevallier

The goal of this paper is to show how different machine learning tools on the
Riemannian manifold $\mathcal{P}_d$ of Symmetric Positive Definite (SPD)
matrices can be united under a probabilistic framework. For this, we will need
several Gaussian distributions defined on $\mathcal{P}_d$. We will show how
popular classifiers on $\mathcal{P}_d$ can be reinterpreted as Bayes
Classifiers using these Gaussian distributions. These distributions will also
be used for outlier detection and dimension reduction. By showing that those
distributions are pervasive in the tools used on $\mathcal{P}_d$, we allow for
other machine learning tools to be extended to $\mathcal{P}_d$.

### 6. A New Approach to Backtracking Counterfactual Explanations: A Causal Framework for Efficient Model Interpretability

[A New Approach to Backtracking Counterfactual Explanations: A Causal Framework for Efficient Model Interpretability](http://arxiv.org/pdf/2505.02435v1)

Authors: Pouria Fatemi, Ehsan Sharifian, Mohammad Hossein Yassaee

Counterfactual explanations enhance interpretability by identifying
alternative inputs that produce different outputs, offering localized insights
into model decisions. However, traditional methods often neglect causal
relationships, leading to unrealistic examples. While newer approaches
integrate causality, they are computationally expensive. To address these
challenges, we propose an efficient method based on backtracking
counterfactuals that incorporates causal reasoning to generate actionable
explanations. We first examine the limitations of existing methods and then
introduce our novel approach and its features. We also explore the relationship
between our method and previous techniques, demonstrating that it generalizes
them in specific scenarios. Finally, experiments show that our method provides
deeper insights into model outputs.

### 7. Resolving Memorization in Empirical Diffusion Model for Manifold Data in High-Dimensional Spaces

[Resolving Memorization in Empirical Diffusion Model for Manifold Data in High-Dimensional Spaces](http://arxiv.org/pdf/2505.02508v1)

Authors: Yang Lyu, Yuchun Qian, Tan Minh Nguyen, Xin T. Tong

Diffusion models is a popular computational tool to generate new data
samples. It utilizes a forward diffusion process that add noise to the data
distribution and then use a reverse process to remove noises to produce samples
from the data distribution. However, when the empirical data distribution
consists of $n$ data point, using the empirical diffusion model will
necessarily produce one of the existing data points. This is often referred to
as the memorization effect, which is usually resolved by sophisticated machine
learning procedures in the current literature. This work shows that the
memorization problem can be resolved by a simple inertia update step at the end
of the empirical diffusion model simulation. Our inertial diffusion model
requires only the empirical diffusion model score function and it does not
require any further training. We show that choosing the inertia diffusion model
sample distribution is an $O\left(n^{-\frac{2}{d+4}}\right)$ Wasserstein-1
approximation of a data distribution lying on a $C^2$ manifold of dimension
$d$. Since this estimate is significant smaller the Wasserstein1 distance
between population and empirical distributions, it rigorously shows the
inertial diffusion model produces new data samples. Remarkably, this upper
bound is completely free of the ambient space dimension, since there is no
training involved. Our analysis utilizes the fact that the inertial diffusion
model samples are approximately distributed as the Gaussian kernel density
estimator on the manifold. This reveals an interesting connection between
diffusion model and manifold learning.

### 8. Advancing Constrained Monotonic Neural Networks: Achieving Universal Approximation Beyond Bounded Activations

[Advancing Constrained Monotonic Neural Networks: Achieving Universal Approximation Beyond Bounded Activations](http://arxiv.org/pdf/2505.02537v1)

Authors: Davide Sartor, Alberto Sinigaglia, Gian Antonio Susto

Conventional techniques for imposing monotonicity in MLPs by construction
involve the use of non-negative weight constraints and bounded activation
functions, which pose well-known optimization challenges. In this work, we
generalize previous theoretical results, showing that MLPs with non-negative
weight constraint and activations that saturate on alternating sides are
universal approximators for monotonic functions. Additionally, we show an
equivalence between the saturation side in the activations and the sign of the
weight constraint. This connection allows us to prove that MLPs with convex
monotone activations and non-positive constrained weights also qualify as
universal approximators, in contrast to their non-negative constrained
counterparts. Our results provide theoretical grounding to the empirical
effectiveness observed in previous works while leading to possible
architectural simplification. Moreover, to further alleviate the optimization
difficulties, we propose an alternative formulation that allows the network to
adjust its activations according to the sign of the weights. This eliminates
the requirement for weight reparameterization, easing initialization and
improving training stability. Experimental evaluation reinforces the validity
of the theoretical results, showing that our novel approach compares favourably
to traditional monotonic architectures.

### 9. Ensemble Kalman filter for uncertainty in human language comprehension

[Ensemble Kalman filter for uncertainty in human language comprehension](http://arxiv.org/pdf/2505.02590v1)

Authors: Diksha Bhandari, Alessandro Lopopolo, Milena Rabovsky, Sebastian Reich

Artificial neural networks (ANNs) are widely used in modeling sentence
processing but often exhibit deterministic behavior, contrasting with human
sentence comprehension, which manages uncertainty during ambiguous or
unexpected inputs. This is exemplified by reversal anomalies-sentences with
unexpected role reversals that challenge syntax and semantics-highlighting the
limitations of traditional ANN models, such as the Sentence Gestalt (SG) Model.
To address these limitations, we propose a Bayesian framework for sentence
comprehension, applying an extension of the ensemble Kalman filter (EnKF) for
Bayesian inference to quantify uncertainty. By framing language comprehension
as a Bayesian inverse problem, this approach enhances the SG model's ability to
reflect human sentence processing with respect to the representation of
uncertainty. Numerical experiments and comparisons with maximum likelihood
estimation (MLE) demonstrate that Bayesian methods improve uncertainty
representation, enabling the model to better approximate human cognitive
processing when dealing with linguistic ambiguities.

### 10. Entropic Mirror Descent for Linear Systems: Polyak's Stepsize and Implicit Bias

[Entropic Mirror Descent for Linear Systems: Polyak's Stepsize and Implicit Bias](http://arxiv.org/pdf/2505.02614v1)

Authors: Yura Malitsky, Alexander Posch

This paper focuses on applying entropic mirror descent to solve linear
systems, where the main challenge for the convergence analysis stems from the
unboundedness of the domain. To overcome this without imposing restrictive
assumptions, we introduce a variant of Polyak-type stepsizes. Along the way, we
strengthen the bound for $\ell_1$-norm implicit bias, obtain sublinear and
linear convergence results, and generalize the convergence result to arbitrary
convex $L$-smooth functions. We also propose an alternative method that avoids
exponentiation, resembling the original Hadamard descent, but with provable
convergence.

