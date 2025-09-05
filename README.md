# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-04 17:00:24.603952 PST.

### Artificial Intelligence

### 1. [Accountability Framework for Healthcare AI Systems: Towards Joint Accountability in Decision Making](http://arxiv.org/pdf/2509.03286v1)

Authors: Prachi Bagave, Marcus Westberg, Marijn Janssen, Aaron Yi Ding

AI is transforming the healthcare domain and is increasingly helping
practitioners to make health-related decisions. Therefore, accountability
becomes a crucial concern for critical AI-driven decisions. Although regulatory
bodies, such as the EU commission, provide guidelines, they are highlevel and
focus on the ''what'' that should be done and less on the ''how'', creating a
knowledge gap for actors. Through an extensive analysis, we found that the term
accountability is perceived and dealt with in many different ways, depending on
the actor's expertise and domain of work. With increasing concerns about AI
accountability issues and the ambiguity around this term, this paper bridges
the gap between the ''what'' and ''how'' of AI accountability, specifically for
AI systems in healthcare. We do this by analysing the concept of
accountability, formulating an accountability framework, and providing a
three-tier structure for handling various accountability mechanisms. Our
accountability framework positions the regulations of healthcare AI systems and
the mechanisms adopted by the actors under a consistent accountability regime.
Moreover, the three-tier structure guides the actors of the healthcare AI
system to categorise the mechanisms based on their conduct. Through our
framework, we advocate that decision-making in healthcare AI holds shared
dependencies, where accountability should be dealt with jointly and should
foster collaborations. We highlight the role of explainability in instigating
communication and information sharing between the actors to further facilitate
the collaborative process.

### 2. [Single Domain Generalization in Diabetic Retinopathy: A Neuro-Symbolic Learning Approach](http://arxiv.org/pdf/2509.02918v1)

Authors: Midhat Urooj, Ayan Banerjee, Farhat Shaikh, Kuntal Thakur, Sandeep Gupta

Domain generalization remains a critical challenge in medical imaging, where
models trained on single sources often fail under real-world distribution
shifts. We propose KG-DG, a neuro-symbolic framework for diabetic retinopathy
(DR) classification that integrates vision transformers with expert-guided
symbolic reasoning to enable robust generalization across unseen domains. Our
approach leverages clinical lesion ontologies through structured, rule-based
features and retinal vessel segmentation, fusing them with deep visual
representations via a confidence-weighted integration strategy. The framework
addresses both single-domain generalization (SDG) and multi-domain
generalization (MDG) by minimizing the KL divergence between domain embeddings,
thereby enforcing alignment of high-level clinical semantics. Extensive
experiments across four public datasets (APTOS, EyePACS, Messidor-1,
Messidor-2) demonstrate significant improvements: up to a 5.2% accuracy gain in
cross-domain settings and a 6% improvement over baseline ViT models. Notably,
our symbolic-only model achieves a 63.67% average accuracy in MDG, while the
complete neuro-symbolic integration achieves the highest accuracy compared to
existing published baselines and benchmarks in challenging SDG scenarios.
Ablation studies reveal that lesion-based features (84.65% accuracy)
substantially outperform purely neural approaches, confirming that symbolic
components act as effective regularizers beyond merely enhancing
interpretability. Our findings establish neuro-symbolic integration as a
promising paradigm for building clinically robust, and domain-invariant medical
AI systems.

### 3. [KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models](http://arxiv.org/pdf/2509.02966v1)

Authors: Yujin Wang, Tianyi Wang, Quanfeng Liu, Wenxian Fan, Junfeng Jiao, Christian Claudel, Yunbing Yan, Bingzhao Gao, Jianqiang Wang, Hong Chen

Accurate short-horizon trajectory prediction is pivotal for safe and reliable
autonomous driving, yet existing vision-language models (VLMs) often fail to
effectively ground their reasoning in scene dynamics and domain knowledge. To
address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM
framework that predicts ego trajectories directly from consecutive front-view
driving frames. KEPT couples a temporal frequency-spatial fusion (TFSF) video
encoder, trained via self-supervised learning with hard-negative mining, with a
scalable k-means + HNSW retrieval stack that supplies scene-aligned exemplars.
Retrieved priors are embedded into chain-of-thought (CoT) prompts with explicit
planning constraints, while a triple-stage fine-tuning schedule incrementally
aligns the language head to metric spatial cues, physically feasible motion,
and temporally conditioned front-view planning. Evaluated on nuScenes dataset,
KEPT achieves state-of-the-art performance across open-loop protocols: under
NoAvg, it achieves 0.70m average L2 with a 0.21\% collision rate; under TemAvg
with lightweight ego status, it attains 0.31m average L2 and a 0.07\% collision
rate. Ablation studies show that all three fine-tuning stages contribute
complementary benefits, and that using Top-2 retrieved exemplars yields the
best accuracy-safety trade-off. The k-means-clustered HNSW index delivers
sub-millisecond retrieval latency, supporting practical deployment. These
results indicate that retrieval-augmented, CoT-guided VLMs offer a promising,
data-efficient pathway toward interpretable and trustworthy autonomous driving.

### 4. [Lesion-Aware Visual-Language Fusion for Automated Image Captioning of Ulcerative Colitis Endoscopic Examinations](http://arxiv.org/pdf/2509.03011v1)

Authors: Alexis Ivan Lopez Escamilla, Gilberto Ochoa, Sharib Al

We present a lesion-aware image captioning framework for ulcerative colitis
(UC). The model integrates ResNet embeddings, Grad-CAM heatmaps, and
CBAM-enhanced attention with a T5 decoder. Clinical metadata (MES score 0-3,
vascular pattern, bleeding, erythema, friability, ulceration) is injected as
natural-language prompts to guide caption generation. The system produces
structured, interpretable descriptions aligned with clinical practice and
provides MES classification and lesion tags. Compared with baselines, our
approach improves caption quality and MES classification accuracy, supporting
reliable endoscopic reporting.

### 5. [Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens](http://arxiv.org/pdf/2509.03025v1)

Authors: Sohee Kim, Soohyun Ryu, Joonhyung Park, Eunho Yang

Large Vision-Language Models (LVLMs) generate contextually relevant responses
by jointly interpreting visual and textual inputs. However, our finding reveals
they often mistakenly perceive text inputs lacking visual evidence as being
part of the image, leading to erroneous responses. In light of this finding, we
probe whether LVLMs possess an internal capability to determine if textual
concepts are grounded in the image, and discover a specific subset of
Feed-Forward Network (FFN) neurons, termed Visual Absence-aware (VA) neurons,
that consistently signal the visual absence through a distinctive activation
pattern. Leveraging these patterns, we develop a detection module that
systematically classifies whether an input token is visually grounded. Guided
by its prediction, we propose a method to refine the outputs by reinterpreting
question prompts or replacing the detected absent tokens during generation.
Extensive experiments show that our method effectively mitigates the models'
tendency to falsely presume the visual presence of text input and its
generality across various LVLMs.

### 6. [MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model](http://arxiv.org/pdf/2509.03041v1)

Authors: Pengyang Yu, Haoquan Wang, Gerard Marks, Tahar Kechadi, Laurence T. Yang, Sahraoui Dhelim, Nyothiri Aung

Accurate skin-lesion segmentation remains a key technical challenge for
computer-aided diagnosis of skin cancer. Convolutional neural networks, while
effective, are constrained by limited receptive fields and thus struggle to
model long-range dependencies. Vision Transformers capture global context, yet
their quadratic complexity and large parameter budgets hinder use on the
small-sample medical datasets common in dermatology. We introduce the
MedLiteNet, a lightweight CNN Transformer hybrid tailored for dermoscopic
segmentation that achieves high precision through hierarchical feature
extraction and multi-scale context aggregation. The encoder stacks depth-wise
Mobile Inverted Bottleneck blocks to curb computation, inserts a
bottleneck-level cross-scale token-mixing unit to exchange information between
resolutions, and embeds a boundary-aware self-attention module to sharpen
lesion contours.

### 7. [FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs](http://arxiv.org/pdf/2509.03047v1)

Authors: Haijun Zhang, Jinxiang Wang, Zhenhua Yu, Yanyong Zhang, Xuejie Ji, Kaining Mao, Jun Zhang, Yaqing Zhang, Ting Wu, Fei Jie, Xiemin Huang, Zhifang Cai, Junhua Cheng, Shuwei Wang, Wei Li, Xiaoming Bao, Hua Xu, Shixiong Zhao, Jun Li, Hongwei Sun, Ziyang Zhang, Yi Xiong, Chunsheng Li

Large language models (LLMs) have made a profound impact across various
fields due to their advanced capabilities. However, training these models at
unprecedented scales requires extensive AI accelerator clusters and
sophisticated parallelism strategies, which pose significant challenges in
maintaining system reliability over prolonged training periods. A major concern
is the substantial loss of training time caused by inevitable hardware and
software failures. To address these challenges, we present FlashRecovery, a
fast and low-cost failure recovery system comprising three core modules: (1)
Active and real-time failure detection. This module performs continuous
training state monitoring, enabling immediate identification of hardware and
software failures within seconds, thus ensuring rapid incident response; (2)
Scale-independent task restart. By employing different recovery strategies for
normal and faulty nodes, combined with an optimized communication group
reconstruction protocol, our approach ensures that the recovery time remains
nearly constant, regardless of cluster scale; (3) Checkpoint-free recovery
within one step. Our novel recovery mechanism enables single-step restoration,
completely eliminating dependence on traditional checkpointing methods and
their associated overhead. Collectively, these innovations enable FlashRecovery
to achieve optimal Recovery Time Objective (RTO) and Recovery Point Objective
(RPO), substantially improving the reliability and efficiency of long-duration
LLM training. Experimental results demonstrate that FlashRecovery system can
achieve training restoration on training cluster with 4, 800 devices in 150
seconds. We also verify that the time required for failure recovery is nearly
consistent for different scales of training tasks.

### 8. [Binary Quantization For LLMs Through Dynamic Grouping](http://arxiv.org/pdf/2509.03054v1)

Authors: Xinzhe Zheng, Zhen-Qun Yang, Haoran Xie, S. Joe Qin, Arlene Chen, Fangzhen Lin

Large Language Models (LLMs) have demonstrated remarkable performance across
a wide range of Natural Language Processing (NLP) tasks, but require
substantial memory and computational resources. Binary quantization, which
compresses model weights from 16-bit Brain Float to 1-bit representations in
{-1, 1}, offers significant reductions in storage and inference costs. However,
such aggressive quantization often leads to notable performance degradation
compared to more conservative 4-bit quantization methods. In this research, we
propose a novel optimization objective tailored for binary quantization, along
with three algorithms designed to realize it effectively. Our method enhances
blocked quantization by dynamically identifying optimal unstructured
sub-matrices through adaptive grouping strategies. Experimental results
demonstrate that our approach achieves an average bit length of just 1.007
bits, while maintaining high model quality. Specifically, our quantized LLaMA
3.2 3B model attains a perplexity of 8.23, remarkably close to the original
7.81, and surpasses previous SOTA BiLLM with a perplexity of only 123.90.
Furthermore, our method is competitive with SOTA 4-bit approaches such as GPTQ
in both performance and efficiency. The compression process is highly
efficient, requiring only 14 seconds to quantize the full LLaMA 3.2 3B weights
on a single CPU core, with the entire process completing in under 100 minutes
and exhibiting embarrassingly parallel properties.
  Code - https://github.com/johnnyzheng0636/WGM_bi_quan

### 9. [Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers](http://arxiv.org/pdf/2509.03059v1)

Authors: Xingyue Huang, Rishabh, Gregor Franke, Ziyi Yang, Jiamu Bai, Weijie Bai, Jinhe Bi, Zifeng Ding, Yiqun Duan, Chengyu Fan, Wendong Fan, Xin Gao, Ruohao Guo, Yuan He, Zhuangzhuang He, Xianglong Hu, Neil Johnson, Bowen Li, Fangru Lin, Siyu Lin, Tong Liu, Yunpu Ma, Hao Shen, Hao Sun, Beibei Wang, Fangyijie Wang, Hao Wang, Haoran Wang, Yang Wang, Yifeng Wang, Zhaowei Wang, Ziyang Wang, Yifan Wu, Zikai Xiao, Chengxing Xie, Fan Yang, Junxiao Yang, Qianshuo Ye, Ziyu Ye, Guangtao Zeng, Yuwen Ebony Zhang, Zeyu Zhang, Zihao Zhu, Bernard Ghanem, Philip Torr, Guohao Li

Recent advances in Large Language Models (LLMs) have shown that their
reasoning capabilities can be significantly improved through Reinforcement
Learning with Verifiable Reward (RLVR), particularly in domains like
mathematics and programming, where ground-truth correctness can be
automatically evaluated. However, extending this success to other
reasoning-intensive domains remains challenging due to the scarcity of
high-quality, verifiable datasets and the high cost of human supervision. In
this work, we introduce the Loong Project: an open-source framework for
scalable synthetic data generation and verification across a diverse range of
reasoning-intensive domains. The framework consists of two key components: (1)
LoongBench, a curated seed dataset containing 8,729 human-vetted examples
across 12 domains (e.g., Advanced Mathematics, Chemistry, Logic), each paired
with executable code and rich metadata; and (2) LoongEnv, a modular synthetic
data generation environment that supports multiple prompting strategies to
produce new question-answer-code triples. Together, these components form an
agent-environment loop that enables reinforcement learning, where an LLM-based
agent is rewarded for generating Chain-of-Thought (CoT) solutions that align
with code-executed answers. Empirically, we benchmark LoongBench on a broad
suite of both open-source and proprietary LLMs to evaluate domain coverage and
reveal performance bottlenecks. In addition, we conduct a comprehensive
analysis of synthetic data generated by LoongEnv, examining correctness,
difficulty, and diversity. Code and documentation are available at
https://github.com/camel-ai/loong.

### 10. [Are We SOLID Yet? An Empirical Study on Prompting LLMs to Detect Design Principle Violations](http://arxiv.org/pdf/2509.03093v1)

Authors: Fatih Pehlivan, Arçin Ülkü Ergüzen, Sahand Moslemi Yengejeh, Mayasah Lami, Anil Koyuncu

Traditional static analysis methods struggle to detect semantic design flaws,
such as violations of the SOLID principles, which require a strong
understanding of object-oriented design patterns and principles. Existing
solutions typically focus on individual SOLID principles or specific
programming languages, leaving a gap in the ability to detect violations across
all five principles in multi-language codebases. This paper presents a new
approach: a methodology that leverages tailored prompt engineering to assess
LLMs on their ability to detect SOLID violations across multiple languages. We
present a benchmark of four leading LLMs-CodeLlama, DeepSeekCoder, QwenCoder,
and GPT-4o Mini-on their ability to detect violations of all five SOLID
principles. For this evaluation, we construct a new benchmark dataset of 240
manually validated code examples. Using this dataset, we test four distinct
prompt strategies inspired by established zero-shot, few-shot, and
chain-of-thought techniques to systematically measure their impact on detection
accuracy. Our emerging results reveal a stark hierarchy among models, with
GPT-4o Mini decisively outperforming others, yet even struggles with
challenging principles like DIP. Crucially, we show that prompt strategy has a
dramatic impact, but no single strategy is universally best; for instance, a
deliberative ENSEMBLE prompt excels at OCP detection while a hint-based EXAMPLE
prompt is superior for DIP violations. Across all experiments, detection
accuracy is heavily influenced by language characteristics and degrades sharply
with increasing code complexity. These initial findings demonstrate that
effective, AI-driven design analysis requires not a single best model, but a
tailored approach that matches the right model and prompt to the specific
design context, highlighting the potential of LLMs to support maintainability
through AI-assisted code analysis.

### Hardware Architecture

### 1. [FastCaps: A Design Methodology for Accelerating Capsule Network on Field Programmable Gate Arrays](http://arxiv.org/pdf/2509.03103v1)

Authors: Abdul Rahoof, Vivek Chaturvedi, Muhammad Shafique

Capsule Network (CapsNet) has shown significant improvement in understanding
the variation in images along with better generalization ability compared to
traditional Convolutional Neural Network (CNN). CapsNet preserves spatial
relationship among extracted features and apply dynamic routing to efficiently
learn the internal connections between capsules. However, due to the capsule
structure and the complexity of the routing mechanism, it is non-trivial to
accelerate CapsNet performance in its original form on Field Programmable Gate
Array (FPGA). Most of the existing works on CapsNet have achieved limited
acceleration as they implement only the dynamic routing algorithm on FPGA,
while considering all the processing steps synergistically is important for
real-world applications of Capsule Networks. Towards this, we propose a novel
two-step approach that deploys a full-fledged CapsNet on FPGA. First, we prune
the network using a novel Look-Ahead Kernel Pruning (LAKP) methodology that
uses the sum of look-ahead scores of the model parameters. Next, we simplify
the nonlinear operations, reorder loops, and parallelize operations of the
routing algorithm to reduce CapsNet hardware complexity. To the best of our
knowledge, this is the first work accelerating a full-fledged CapsNet on FPGA.
Experimental results on the MNIST and F-MNIST datasets (typical in Capsule
Network community) show that the proposed LAKP approach achieves an effective
compression rate of 99.26% and 98.84%, and achieves a throughput of 82 FPS and
48 FPS on Xilinx PYNQ-Z1 FPGA, respectively. Furthermore, reducing the hardware
complexity of the routing algorithm increases the throughput to 1351 FPS and
934 FPS respectively. As corroborated by our results, this work enables highly
performance-efficient deployment of CapsNets on low-cost FPGA that are popular
in modern edge devices.

### 2. [CapsBeam: Accelerating Capsule Network based Beamformer for Ultrasound Non-Steered Plane Wave Imaging on Field Programmable Gate Array](http://arxiv.org/pdf/2509.03201v1)

Authors: Abdul Rahoof, Vivek Chaturvedi, Mahesh Raveendranatha Panicker, Muhammad Shafique

In recent years, there has been a growing trend in accelerating
computationally complex non-real-time beamforming algorithms in ultrasound
imaging using deep learning models. However, due to the large size and
complexity these state-of-the-art deep learning techniques poses significant
challenges when deploying on resource-constrained edge devices. In this work,
we propose a novel capsule network based beamformer called CapsBeam, designed
to operate on raw radio-frequency data and provide an envelope of beamformed
data through non-steered plane wave insonification. Experiments on in-vivo
data, CapsBeam reduced artifacts compared to the standard Delay-and-Sum (DAS)
beamforming. For in-vitro data, CapsBeam demonstrated a 32.31% increase in
contrast, along with gains of 16.54% and 6.7% in axial and lateral resolution
compared to the DAS. Similarly, in-silico data showed a 26% enhancement in
contrast, along with improvements of 13.6% and 21.5% in axial and lateral
resolution, respectively, compared to the DAS. To reduce the parameter
redundancy and enhance the computational efficiency, we pruned the model using
our multi-layer LookAhead Kernel Pruning (LAKP-ML) methodology, achieving a
compression ratio of 85% without affecting the image quality. Additionally, the
hardware complexity of the proposed model is reduced by applying quantization,
simplification of non-linear operations, and parallelizing operations. Finally,
we proposed a specialized accelerator architecture for the pruned and optimized
CapsBeam model, implemented on a Xilinx ZU7EV FPGA. The proposed accelerator
achieved a throughput of 30 GOPS for the convolution operation and 17.4 GOPS
for the dynamic routing operation.

### 3. [Amplifying Effective CXL Memory Bandwidth for LLM Inference via Transparent Near-Data Processing](http://arxiv.org/pdf/2509.03377v1)

Authors: Rui Xie, Asad Ul Haq, Linsen Ma, Yunhua Fang, Zirak Burzin Engineer, Liu Liu, Tong Zhang

Large language model (LLM) inference is bottlenecked by the limited bandwidth
of CXL-based memory used for capacity expansion. We introduce CXL-NDP, a
transparent near-data processing architecture that amplifies effective CXL
bandwidth without requiring changes to the CXL.mem interface or AI models.
CXL-NDP integrates a precision-scalable bit-plane layout for dynamic
quantization with transparent lossless compression of weights and KV caches
directly within the CXL device. In end-to-end serving, CXL-NDP improves
throughput by 43%, extends the maximum context length by 87%, and reduces the
KV cache footprint by 46.9% without accuracy loss. Hardware synthesis confirms
its practicality with a modest silicon footprint, lowering the barrier for
adopting efficient, scalable CXL-based memory in generative AI infrastructure.

### Computational Complexity

### 1. [Information-Theoretic Lower Bounds for Approximating Monomials via Optimal Quantum Tsallis Entropy Estimation](http://arxiv.org/pdf/2509.03496v1)

Authors: Qisheng Wang

This paper reveals a conceptually new connection from information theory to
approximation theory via quantum algorithms for entropy estimation.
Specifically, we provide an information-theoretic lower bound
$\Omega(\sqrt{n})$ on the approximate degree of the monomial $x^n$, compared to
the analytic lower bounds shown in Newman and Rivlin (Aequ. Math. 1976) via
Fourier analysis and in Sachdeva and Vishnoi (Found. Trends Theor. Comput. Sci.
2014) via the Markov brothers' inequality. This is done by relating the
polynomial approximation of monomials to quantum Tsallis entropy estimation.
This further implies a quantum algorithm that estimates to within additive
error $\varepsilon$ the Tsallis entropy of integer order $q \geq 2$ of an
unknown probability distribution $p$ or an unknown quantum state $\rho$, using
$\widetilde \Theta(\frac{1}{\sqrt{q}\varepsilon})$ queries to the quantum
oracle that produces a sample from $p$ or prepares a copy of $\rho$, improving
the prior best $O(\frac{1}{\varepsilon})$ via the Shift test due to Ekert,
Alves, Oi, Horodecki, Horodecki and Kwek (Phys. Rev. Lett. 2002). To the best
of our knowledge, this is the first quantum entropy estimator with optimal
query complexity (up to polylogarithmic factors) for all parameters
simultaneously.

### Computational Engineering

### 1. [Automatic Differentiation of Agent-Based Models](http://arxiv.org/pdf/2509.03303v1)

Authors: Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu, Doyne Farmer, Michael Wooldridge

Agent-based models (ABMs) simulate complex systems by capturing the bottom-up
interactions of individual agents comprising the system. Many complex systems
of interest, such as epidemics or financial markets, involve thousands or even
millions of agents. Consequently, ABMs often become computationally demanding
and rely on the calibration of numerous free parameters, which has
significantly hindered their widespread adoption. In this paper, we demonstrate
that automatic differentiation (AD) techniques can effectively alleviate these
computational burdens. By applying AD to ABMs, the gradients of the simulator
become readily available, greatly facilitating essential tasks such as
calibration and sensitivity analysis. Specifically, we show how AD enables
variational inference (VI) techniques for efficient parameter calibration. Our
experiments demonstrate substantial performance improvements and computational
savings using VI on three prominent ABMs: Axtell's model of firms; Sugarscape;
and the SIR epidemiological model. Our approach thus significantly enhances the
practicality and scalability of ABMs for studying complex systems.

### 2. [Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](http://arxiv.org/pdf/2509.03340v1)

Authors: Fleur Hendriks, Ondřej Rokoš, Martin Doškář, Marc G. D. Geers, Vlado Menkovski

Bifurcation phenomena in nonlinear dynamical systems often lead to multiple
coexisting stable solutions, particularly in the presence of symmetry breaking.
Deterministic machine learning models struggle to capture this multiplicity,
averaging over solutions and failing to represent lower-symmetry outcomes. In
this work, we propose a generative framework based on flow matching to model
the full probability distribution over bifurcation outcomes. Our method enables
direct sampling of multiple valid solutions while preserving system symmetries
through equivariant modeling. We introduce a symmetric matching strategy that
aligns predicted and target outputs under group actions, allowing accurate
learning in equivariant settings. We validate our approach on a range of
systems, from toy models to complex physical problems such as buckling beams
and the Allen-Cahn equation. Our results demonstrate that flow matching
significantly outperforms non-probabilistic and variational methods in
capturing multimodal distributions and symmetry-breaking bifurcations, offering
a principled and scalable solution for modeling multistability in
high-dimensional systems.

### Computational Geometry

### 1. [Triangle Detection in Worst-Case Sparse Graphs via Local Sketching](http://arxiv.org/pdf/2509.03215v1)

Authors: Hongyi Duan, Jian'an Zhang

We present a non-algebraic, locality-preserving framework for triangle
detection in worst-case sparse graphs. Our algorithm processes the graph in
$O(\log n)$ independent layers and partitions incident edges into prefix-based
classes where each class maintains a 1-sparse triple over a prime field.
Potential witnesses are surfaced by pair-key (PK) alignment, and every
candidate is verified by a three-stage, zero-false-positive pipeline: a
class-level 1-sparse consistency check, two slot-level decodings, and a final
adjacency confirmation. \textbf{To obtain single-run high-probability coverage,
we further instantiate $R=c_G\log n$ independent PK groups per class (each
probing a constant number of complementary buckets), which amplifies the
per-layer hit rate from $\Theta(1/\log n)$ to $1-n^{-\Omega(1)}$ without
changing the accounting.} A one-shot pairing discipline and class-term
triggering yield a per-(layer,level) accounting bound of $O(m)$, while
keep-coin concentration ensures that each vertex retains only $O(d^+(x))$ keys
with high probability. Consequently, the total running time is $O(m\log^2 n)$
and the peak space is $O(m\log n)$, both with high probability. The algorithm
emits a succinct Seeds+Logs artifact that enables a third party to replay all
necessary checks and certify a NO-instance in $\tilde O(m\log n)$ time. We also
prove a $\Theta(1/\log n)$ hit-rate lower bound for any single PK family under
a constant-probe local model (via Yao)--motivating the use of $\Theta(\log n)$
independent groups--and discuss why global algebraic convolutions would break
near-linear accounting or run into fine-grained barriers. We outline measured
paths toward Las Vegas $O(m\log n)$ and deterministic near-linear variants.

### Computation and Language

### 1. [Advancing Minority Stress Detection with Transformers: Insights from the Social Media Datasets](http://arxiv.org/pdf/2509.02908v1)

Authors: Santosh Chapagain, Cory J Cascalheira, Shah Muhammad Hamdi, Soukaina Filali Boubrahimi, Jillian R. Scheer

Individuals from sexual and gender minority groups experience
disproportionately high rates of poor health outcomes and mental disorders
compared to their heterosexual and cisgender counterparts, largely as a
consequence of minority stress as described by Meyer's (2003) model. This study
presents the first comprehensive evaluation of transformer-based architectures
for detecting minority stress in online discourse. We benchmark multiple
transformer models including ELECTRA, BERT, RoBERTa, and BART against
traditional machine learning baselines and graph-augmented variants. We further
assess zero-shot and few-shot learning paradigms to assess their applicability
on underrepresented datasets. Experiments are conducted on the two largest
publicly available Reddit corpora for minority stress detection, comprising
12,645 and 5,789 posts, and are repeated over five random seeds to ensure
robustness. Our results demonstrate that integrating graph structure
consistently improves detection performance across transformer-only models and
that supervised fine-tuning with relational context outperforms zero and
few-shot approaches. Theoretical analysis reveals that modeling social
connectivity and conversational context via graph augmentation sharpens the
models' ability to identify key linguistic markers such as identity
concealment, internalized stigma, and calls for support, suggesting that
graph-enhanced transformers offer the most reliable foundation for digital
health interventions and public health policy.

### 2. [English Pronunciation Evaluation without Complex Joint Training: LoRA Fine-tuned Speech Multimodal LLM](http://arxiv.org/pdf/2509.02915v1)

Authors: Taekyung Ahn, Hosung Nam

This study demonstrates that a Multimodal Large Language Model (MLLM) adapted
via Low-Rank Adaptation (LoRA) can perform both Automatic Pronunciation
Assessment (APA) and Mispronunciation Detection and Diagnosis (MDD)
simultaneously. Leveraging Microsoft's Phi-4-multimodal-instruct, our
fine-tuning method eliminates the need for complex architectural changes or
separate training procedures conventionally required for these distinct tasks.
Fine-tuned on the Speechocean762 dataset, the pronunciation evaluation scores
predicted by the model exhibited a strong Pearson Correlation Coefficient (PCC
> 0.7) with human-assigned scores, while achieving low Word Error Rate (WER)
and Phoneme Error Rate (PER) (both < 0.15). Notably, fine-tuning only the LoRA
layers was sufficient to achieve performance levels comparable to those
achieved by fine-tuning all audio layers. This research highlights that an
integrated pronunciation assessment system can be established by adapting large
multimodal models without full fine-tuning, utilizing a significantly simpler
training methodology compared to previous joint models designed for
simultaneous APA and MDD. This efficient LoRA-based approach paves the way for
more accessible, integrated, and effective Computer-Assisted Pronunciation
Training (CAPT) technologies for English L2 learners.

### 3. [Decoding the Rule Book: Extracting Hidden Moderation Criteria from Reddit Communities](http://arxiv.org/pdf/2509.02926v1)

Authors: Youngwoo Kim, Himanshu Beniwal, Steven L. Johnson, Thomas Hartvigsen

Effective content moderation systems require explicit classification
criteria, yet online communities like subreddits often operate with diverse,
implicit standards. This work introduces a novel approach to identify and
extract these implicit criteria from historical moderation data using an
interpretable architecture. We represent moderation criteria as score tables of
lexical expressions associated with content removal, enabling systematic
comparison across different communities. Our experiments demonstrate that these
extracted lexical patterns effectively replicate the performance of neural
moderation models while providing transparent insights into decision-making
processes. The resulting criteria matrix reveals significant variations in how
seemingly shared norms are actually enforced, uncovering previously
undocumented moderation patterns including community-specific tolerances for
language, features for topical restrictions, and underlying subcategories of
the toxic speech classification.

### 4. [DiaCBT: A Long-Periodic Dialogue Corpus Guided by Cognitive Conceptualization Diagram for CBT-based Psychological Counseling](http://arxiv.org/pdf/2509.02999v1)

Authors: Yougen Zhou, Ningning Zhou, Qin Chen, Jie Zhou, Aimin Zhou, Liang He

Psychotherapy reaches only a small fraction of individuals suffering from
mental disorders due to social stigma and the limited availability of
therapists. Large language models (LLMs), when equipped with professional
psychotherapeutic skills, offer a promising solution to expand access to mental
health services. However, the lack of psychological conversation datasets
presents significant challenges in developing effective psychotherapy-guided
conversational agents. In this paper, we construct a long-periodic dialogue
corpus for counseling based on cognitive behavioral therapy (CBT). Our curated
dataset includes multiple sessions for each counseling and incorporates
cognitive conceptualization diagrams (CCDs) to guide client simulation across
diverse scenarios. To evaluate the utility of our dataset, we train an in-depth
counseling model and present a comprehensive evaluation framework to benchmark
it against established psychological criteria for CBT-based counseling. Results
demonstrate that DiaCBT effectively enhances LLMs' ability to emulate
psychologists with CBT expertise, underscoring its potential for training more
professional counseling agents.

### 5. [Structure-Learnable Adapter Fine-Tuning for Parameter-Efficient Large Language Models](http://arxiv.org/pdf/2509.03057v1)

Authors: Ming Gong, Yingnan Deng, Nia Qi, Yujun Zou, Zhihao Xue, Yun Zi

This paper addresses the issues of parameter redundancy, rigid structure, and
limited task adaptability in the fine-tuning of large language models. It
proposes an adapter-based fine-tuning method built on a structure-learnable
mechanism. By introducing differentiable gating functions and structural
sparsity control variables, the method enables automatic optimization of
adapter insertion points, activation paths, and module combinations. This
allows the model to adjust its structure flexibly in multi-task settings to
match different task characteristics. With the backbone parameters kept frozen,
the method uses a structure search mechanism to guide the dynamic construction
of task-specific efficient substructures during training. This significantly
improves parameter utilization and representational capacity. In addition, the
paper designs a set of sensitivity analysis experiments to systematically
evaluate the effects of sparsity weight, noise injection ratio, and data
perturbation on model performance. These experiments verify the stability and
robustness of the proposed method across various multi-task natural language
understanding tasks. The experimental results show that the proposed method
outperforms mainstream parameter-efficient tuning techniques on multiple tasks.
It achieves a better balance among accuracy, compression rate, and robustness
to noise and perturbation.

### 6. [A Long Short-Term Memory (LSTM) Model for Business Sentiment Analysis Based on Recurrent Neural Network](http://arxiv.org/pdf/2509.03060v1)

Authors: Md. Jahidul Islam Razin, Md. Abdul Karim, M. F. Mridha, S M Rafiuddin, Tahira Alam

Business sentiment analysis (BSA) is one of the significant and popular
topics of natural language processing. It is one kind of sentiment analysis
techniques for business purposes. Different categories of sentiment analysis
techniques like lexicon-based techniques and different types of machine
learning algorithms are applied for sentiment analysis on different languages
like English, Hindi, Spanish, etc. In this paper, long short-term memory (LSTM)
is applied for business sentiment analysis, where a recurrent neural network is
used. An LSTM model is used in a modified approach to prevent the vanishing
gradient problem rather than applying the conventional recurrent neural network
(RNN). To apply the modified RNN model, product review dataset is used. In this
experiment, 70\% of the data is trained for the LSTM and the rest 30\% of the
data is used for testing. The result of this modified RNN model is compared
with other conventional RNN models, and a comparison is made among the results.
It is noted that the proposed model performs better than the other conventional
RNN models. Here, the proposed model, i.e., the modified RNN model approach has
achieved around 91.33\% of accuracy. By applying this model, any business
company or e-commerce business site can identify the feedback from their
customers about different types of products that customers like or dislike.
Based on the customer reviews, a business company or e-commerce platform can
evaluate its marketing strategy.

### 7. [Measuring Scalar Constructs in Social Science with LLMs](http://arxiv.org/pdf/2509.03116v1)

Authors: Hauke Licht, Rupak Sarkar, Patrick Y. Wu, Pranav Goel, Niklas Stoehr, Elliott Ash, Alexander Miserlis Hoyle

Many constructs that characterize language, like its complexity or
emotionality, have a naturally continuous semantic structure; a public speech
is not just "simple" or "complex," but exists on a continuum between extremes.
Although large language models (LLMs) are an attractive tool for measuring
scalar constructs, their idiosyncratic treatment of numerical outputs raises
questions of how to best apply them. We address these questions with a
comprehensive evaluation of LLM-based approaches to scalar construct
measurement in social science. Using multiple datasets sourced from the
political science literature, we evaluate four approaches: unweighted direct
pointwise scoring, aggregation of pairwise comparisons,
token-probability-weighted pointwise scoring, and finetuning. Our study yields
actionable findings for applied researchers. First, LLMs prompted to generate
pointwise scores directly from texts produce discontinuous distributions with
bunching at arbitrary numbers. The quality of the measurements improves with
pairwise comparisons made by LLMs, but it improves even more by taking
pointwise scores and weighting them by token probability. Finally, finetuning
smaller models with as few as 1,000 training pairs can match or exceed the
performance of prompted LLMs.

### 8. [An experimental and computational study of an Estonian single-person word naming](http://arxiv.org/pdf/2509.03143v1)

Authors: Kaidi Lõo, Arvi Tavast, Maria Heitmeier, Harald Baayen

This study investigates lexical processing in Estonian. A large-scale
single-subject experiment is reported that combines the word naming task with
eye-tracking. Five response variables (first fixation duration, total fixation
duration, number of fixations, word naming latency, and spoken word duration)
are analyzed with the generalized additive model. Of central interest is the
question of whether measures for lexical processing generated by a
computational model of the mental lexicon (the Discriminative Lexicon Model,
DLM) are predictive for these response variables, and how they compare to
classical predictors such as word frequency, neighborhood size, and
inflectional paradigm size. Computational models were implemented both with
linear and deep mappings. Central findings are, first, that DLM-based measures
are powerful predictors for lexical processing, second, that DLM-measures using
deep learning are not necessarily more precise predictors of lexical processing
than DLM-measures using linear mappings, third, that classical predictors tend
to provide somewhat more precise fits compared to DLM-based predictors (except
for total fixation duration, where the two provide equivalent goodness of fit),
and fourth, that in the naming task lexical variables are not predictive for
first fixation duration and the total number of fixations. As the DLM works
with mappings from form to meaning, the predictivity of DLM-based measures for
total fixation duration, naming latencies, and spoken word duration indicates
that meaning is heavily involved in the present word naming task.

### 9. [Expanding the WMT24++ Benchmark with Rumantsch Grischun, Sursilvan, Sutsilvan, Surmiran, Puter, and Vallader](http://arxiv.org/pdf/2509.03148v1)

Authors: Jannis Vamvas, Ignacio Pérez Prat, Not Battesta Soliva, Sandra Baltermia-Guetg, Andrina Beeli, Simona Beeli, Madlaina Capeder, Laura Decurtins, Gian Peder Gregori, Flavia Hobi, Gabriela Holderegger, Arina Lazzarini, Viviana Lazzarini, Walter Rosselli, Bettina Vital, Anna Rutkiewicz, Rico Sennrich

The Romansh language, spoken in Switzerland, has limited resources for
machine translation evaluation. In this paper, we present a benchmark for six
varieties of Romansh: Rumantsch Grischun, a supra-regional variety, and five
regional varieties: Sursilvan, Sutsilvan, Surmiran, Puter, and Vallader. Our
reference translations were created by human translators based on the WMT24++
benchmark, which ensures parallelism with more than 55 other languages. An
automatic evaluation of existing MT systems and LLMs shows that translation out
of Romansh into German is handled relatively well for all the varieties, but
translation into Romansh is still challenging.

### 10. [SinhalaMMLU: A Comprehensive Benchmark for Evaluating Multitask Language Understanding in Sinhala](http://arxiv.org/pdf/2509.03162v1)

Authors: Ashmari Pramodya, Nirasha Nelki, Heshan Shalinda, Chamila Liyanage, Yusuke Sakai, Randil Pushpananda, Ruvan Weerasinghe, Hidetaka Kamigaito, Taro Watanabe

Large Language Models (LLMs) demonstrate impressive general knowledge and
reasoning abilities, yet their evaluation has predominantly focused on global
or anglocentric subjects, often neglecting low-resource languages and
culturally specific content. While recent multilingual benchmarks attempt to
bridge this gap, many rely on automatic translation, which can introduce errors
and misrepresent the original cultural context. To address this, we introduce
SinhalaMMLU, the first multiple-choice question answering benchmark designed
specifically for Sinhala, a low-resource language. The dataset includes over
7,000 questions spanning secondary to collegiate education levels, aligned with
the Sri Lankan national curriculum, and covers six domains and 30 subjects,
encompassing both general academic topics and culturally grounded knowledge. We
evaluate 26 LLMs on SinhalaMMLU and observe that, while Claude 3.5 sonnet and
GPT-4o achieve the highest average accuracies at 67% and 62% respectively,
overall model performance remains limited. In particular, models struggle in
culturally rich domains such as the Humanities, revealing substantial room for
improvement in adapting LLMs to low-resource and culturally specific contexts.

### Cryptography and Security

### 1. [EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint](http://arxiv.org/pdf/2509.03058v1)

Authors: Zhenhua Xu, Meng Han, Wenpeng Xing

The proliferation of large language models (LLMs) has intensified concerns
over model theft and license violations, necessitating robust and stealthy
ownership verification. Existing fingerprinting methods either require
impractical white-box access or introduce detectable statistical anomalies. We
propose EverTracer, a novel gray-box fingerprinting framework that ensures
stealthy and robust model provenance tracing. EverTracer is the first to
repurpose Membership Inference Attacks (MIAs) for defensive use, embedding
ownership signals via memorization instead of artificial trigger-output
overfitting. It consists of Fingerprint Injection, which fine-tunes the model
on any natural language data without detectable artifacts, and Verification,
which leverages calibrated probability variation signal to distinguish
fingerprinted models. This approach remains robust against adaptive
adversaries, including input level modification, and model-level modifications.
Extensive experiments across architectures demonstrate EverTracer's
state-of-the-art effectiveness, stealthness, and resilience, establishing it as
a practical solution for securing LLM intellectual property. Our code and data
are publicly available at https://github.com/Xuzhenhua55/EverTracer.

### 2. [Compressed verification for post-quantum signatures with long-term public keys](http://arxiv.org/pdf/2509.03098v1)

Authors: Gustavo Banegas, Anaëlle Le Dévéhat, Benjamin Smith

Many signature applications-such as root certificates, secure software
updates, and authentication protocols-involve long-lived public keys that are
transferred or installed once and then used for many verifications. This key
longevity makes post-quantum signature schemes with conservative assumptions
(e.g., structure-free lattices) attractive for long-term security. But many
such schemes, especially those with short signatures, suffer from extremely
large public keys. Even in scenarios where bandwidth is not a major concern,
large keys increase storage costs and slow down verification. We address this
with a method to replace large public keys in GPV-style signatures with
smaller, private verification keys. This significantly reduces verifier storage
and runtime while preserving security. Applied to the conservative,
short-signature schemes Wave and Squirrels, our method compresses Squirrels-I
keys from 665 kB to 20.7 kB and Wave822 keys from 3.5 MB to 207.97 kB.

### 3. [PromptCOS: Towards System Prompt Copyright Auditing for LLMs via Content-level Output Similarity](http://arxiv.org/pdf/2509.03117v1)

Authors: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Bingrun Yang, Zhibo Wang, Dacheng Tao, Zhan Qin

The rapid progress of large language models (LLMs) has greatly enhanced
reasoning tasks and facilitated the development of LLM-based applications. A
critical factor in improving LLM-based applications is the design of effective
system prompts, which significantly impact the behavior and output quality of
LLMs. However, system prompts are susceptible to theft and misuse, which could
undermine the interests of prompt owners. Existing methods protect prompt
copyrights through watermark injection and verification but face challenges due
to their reliance on intermediate LLM outputs (e.g., logits), which limits
their practical feasibility.
  In this paper, we propose PromptCOS, a method for auditing prompt copyright
based on content-level output similarity. It embeds watermarks by optimizing
the prompt while simultaneously co-optimizing a special verification query and
content-level signal marks. This is achieved by leveraging cyclic output
signals and injecting auxiliary tokens to ensure reliable auditing in
content-only scenarios. Additionally, it incorporates cover tokens to protect
the watermark from malicious deletion. For copyright verification, PromptCOS
identifies unauthorized usage by comparing the similarity between the
suspicious output and the signal mark. Experimental results demonstrate that
our method achieves high effectiveness (99.3% average watermark similarity),
strong distinctiveness (60.8% greater than the best baseline), high fidelity
(accuracy degradation of no more than 0.58%), robustness (resilience against
three types of potential attacks), and computational efficiency (up to 98.1%
reduction in computational cost). Our code is available at GitHub
https://github.com/LianPing-cyber/PromptCOS.

### 4. [Kangaroo: A Private and Amortized Inference Framework over WAN for Large-Scale Decision Tree Evaluation](http://arxiv.org/pdf/2509.03123v1)

Authors: Wei Xu, Hui Zhu, Yandong Zheng, Song Bian, Ning Sun, Hao Yuan, Dengguo Feng, Hui Li

With the rapid adoption of Models-as-a-Service, concerns about data and model
privacy have become increasingly critical. To solve these problems, various
privacy-preserving inference schemes have been proposed. In particular, due to
the efficiency and interpretability of decision trees, private decision tree
evaluation (PDTE) has garnered significant attention. However, existing PDTE
schemes suffer from significant limitations: their communication and
computation costs scale with the number of trees, the number of nodes, or the
tree depth, which makes them inefficient for large-scale models, especially
over WAN networks. To address these issues, we propose Kangaroo, a private and
amortized decision tree inference framework build upon packed homomorphic
encryption. Specifically, we design a novel model hiding and encoding scheme,
together with secure feature selection, oblivious comparison, and secure path
evaluation protocols, enabling full amortization of the overhead as the number
of nodes or trees scales. Furthermore, we enhance the performance and
functionality of the framework through optimizations, including
same-sharing-for-same-model, latency-aware, and adaptive encoding adjustment
strategies. Kangaroo achieves a $14\times$ to $59\times$ performance
improvement over state-of-the-art (SOTA) one-round interactive schemes in WAN
environments. For large-scale decision tree inference tasks, it delivers a
$3\times$ to $44\times$ speedup compared to existing schemes. Notably, Kangaroo
enables the evaluation of a random forest with $969$ trees and $411825$ nodes
in approximately $60$ ms per tree (amortized) under WAN environments.

### 5. [Exposing Privacy Risks in Anonymizing Clinical Data: Combinatorial Refinement Attacks on k-Anonymity Without Auxiliary Information](http://arxiv.org/pdf/2509.03350v1)

Authors: Somiya Chhillar, Mary K. Righi, Rebecca E. Sutter, Evgenios M. Kornaropoulos

Despite longstanding criticism from the privacy community, k-anonymity
remains a widely used standard for data anonymization, mainly due to its
simplicity, regulatory alignment, and preservation of data utility. However,
non-experts often defend k-anonymity on the grounds that, in the absence of
auxiliary information, no known attacks can compromise its protections. In this
work, we refute this claim by introducing Combinatorial Refinement Attacks
(CRA), a new class of privacy attacks targeting k-anonymized datasets produced
using local recoding. This is the first method that does not rely on external
auxiliary information or assumptions about the underlying data distribution.
CRA leverages the utility-optimizing behavior of local recoding anonymization
of ARX, which is a widely used open-source software for anonymizing data in
clinical settings, to formulate a linear program that significantly reduces the
space of plausible sensitive values. To validate our findings, we partnered
with a network of free community health clinics, an environment where (1)
auxiliary information is indeed hard to find due to the population they serve
and (2) open-source k-anonymity solutions are attractive due to regulatory
obligations and limited resources. Our results on real-world clinical microdata
reveal that even in the absence of external information, established
anonymization frameworks do not deliver the promised level of privacy, raising
critical privacy concerns.

### 6. [Federated Learning: An approach with Hybrid Homomorphic Encryption](http://arxiv.org/pdf/2509.03427v1)

Authors: Pedro Correia, Ivan Silva, Ivone Amorim, Eva Maia, Isabel Praça

Federated Learning (FL) is a distributed machine learning approach that
promises privacy by keeping the data on the device. However, gradient
reconstruction and membership-inference attacks show that model updates still
leak information. Fully Homomorphic Encryption (FHE) can address those privacy
concerns but it suffers from ciphertext expansion and requires prohibitive
overhead on resource-constrained devices. We propose the first Hybrid
Homomorphic Encryption (HHE) framework for FL that pairs the PASTA symmetric
cipher with the BFV FHE scheme. Clients encrypt local model updates with PASTA
and send both the lightweight ciphertexts and the PASTA key (itself
BFV-encrypted) to the server, which performs a homomorphic evaluation of the
decryption circuit of PASTA and aggregates the resulting BFV ciphertexts. A
prototype implementation, developed on top of the Flower FL framework, shows
that on independently and identically distributed MNIST dataset with 12 clients
and 10 training rounds, the proposed HHE system achieves 97.6% accuracy, just
1.3% below plaintext, while reducing client upload bandwidth by over 2,000x and
cutting client runtime by 30% compared to a system based solely on the BFV FHE
scheme. However, server computational cost increases by roughly 15621x for each
client participating in the training phase, a challenge to be addressed in
future work.

### 7. [Evaluating Diverse Feature Extraction Techniques of Multifaceted IoT Malware Analysis: A Survey](http://arxiv.org/pdf/2509.03442v1)

Authors: Zhuoyun Qian, Hongyi Miao, Yili Jiang, Qin Hu, Jiaqi Huang, Cheng Zhang, Fangtian Zhong

As IoT devices continue to proliferate, their reliability is increasingly
constrained by security concerns. In response, researchers have developed
diverse malware analysis techniques to detect and classify IoT malware. These
techniques typically rely on extracting features at different levels from IoT
applications, giving rise to a wide range of feature extraction methods.
However, current approaches still face significant challenges when applied in
practice. This survey provides a comprehensive review of feature extraction
techniques for IoT malware analysis from multiple perspectives. We first
examine static and dynamic feature extraction methods, followed by hybrid
approaches. We then explore feature representation strategies based on graph
learning. Finally, we compare the strengths and limitations of existing
techniques, highlight open challenges, and outline promising directions for
future research.

### 8. [Integrating Generative AI into Cybersecurity Education: A Study of OCR and Multimodal LLM-assisted Instruction](http://arxiv.org/pdf/2509.02998v1)

Authors: Karan Patel, Yu-Zheng Lin, Gaurangi Raul, Bono Po-Jen Shih, Matthew W. Redondo, Banafsheh Saber Latibari, Jesus Pacheco, Soheil Salehi, Pratik Satam

This full paper describes an LLM-assisted instruction integrated with a
virtual cybersecurity lab platform. The digital transformation of Fourth
Industrial Revolution (4IR) systems is reshaping workforce needs, widening
skill gaps, especially among older workers. With rising emphasis on robotics,
automation, AI, and security, re-skilling and up-skilling are essential.
Generative AI can help build this workforce by acting as an instructional
assistant to support skill acquisition during experiential learning. We present
a generative AI instructional assistant integrated into a prior experiential
learning platform. The assistant employs a zero-shot OCR-LLM pipeline within
the legacy Cybersecurity Labs-as-a-Service (CLaaS) platform (2015). Text is
extracted from slide images using Tesseract OCR, then simplified instructions
are generated via a general-purpose LLM, enabling real-time instructional
support with minimal infrastructure. The system was evaluated in a live
university course where student feedback (n=42) averaged 7.83/10, indicating
strong perceived usefulness. A comparative study with multimodal LLMs that
directly interpret slide images showed higher performance on visually dense
slides, but the OCR-LLM pipeline provided comparable pedagogical value on
text-centric slides with much lower computational overhead and cost. This work
demonstrates that a lightweight, easily integrable pipeline can effectively
extend legacy platforms with modern generative AI, offering scalable
enhancements for student comprehension in technical education.

### 9. [Closing the Visibility Gap: A Monitoring Framework for Verifiable Open RAN Operations](http://arxiv.org/pdf/2509.03000v1)

Authors: Hexuan Yu, Md Mohaimin Al Barat, Yang Xiao, Y. Thomas Hou, Wenjing Lou

Open Radio Access Network (Open RAN) is reshaping mobile network architecture
by promoting openness, disaggregation, and cross-vendor interoperability.
However, this architectural flexibility introduces new security challenges,
especially in deployments where multiple mobile network operators (MNOs)
jointly operate shared components. Existing Zero Trust Architectures (ZTA) in
O-RAN, as defined by governmental and industry standards, implicitly assume
that authenticated components will comply with operational policies. However,
this assumption creates a critical blind spot: misconfigured or compromised
components can silently violate policies, misuse resources, or corrupt
downstream processes (e.g., ML-based RIC xApps).
  To address this critical gap, we propose a monitoring framework for low-trust
O-RAN environments that proactively verifies configuration state and control
behavior against tenant-defined policies. Our system provides scalable,
verifiable oversight to enhance transparency and trust in O-RAN operations. We
implement and evaluate the framework using standardized O-RAN configurations,
with total processing latency of approximately 200 ms, demonstrating its
efficiency and practicality for timely policy enforcement and compliance
auditing in multi-MNO deployments.

### 10. [Evaluating Security Properties in the Execution of Quantum Circuits](http://arxiv.org/pdf/2509.03306v1)

Authors: Paolo Bernardi, Antonio Brogi, Gian-Luigi Ferrari, Giuseppe Bisicchia

Quantum computing is a disruptive technology that is expected to offer
significant advantages in many critical fields (e.g. drug discovery and
cryptography). The security of information processed by such machines is
therefore paramount. Currently, modest Noisy Intermediate-Scale Quantum (NISQ)
devices are available. The goal of this work is to identify a practical,
heuristic methodology to evaluate security properties, such as secrecy and
integrity, while using quantum processors owned by potentially untrustworthy
providers.

### Computer Vision and Pattern Recognition

### 1. [LiGuard: A Streamlined Open-Source Framework for Rapid & Interactive Lidar Research](http://arxiv.org/pdf/2509.02902v1)

Authors: Muhammad Shahbaz, Shaurya Agarwal

There is a growing interest in the development of lidar-based autonomous
mobility and Intelligent Transportation Systems (ITS). To operate and research
on lidar data, researchers often develop code specific to application niche.
This approach leads to duplication of efforts across studies that, in many
cases, share multiple methodological steps such as data input/output (I/O),
pre/post processing, and common algorithms in multi-stage solutions. Moreover,
slight changes in data, algorithms, and/or research focus may force major
revisions in the code. To address these challenges, we present LiGuard, an
open-source software framework that allows researchers to: 1) rapidly develop
code for their lidar-based projects by providing built-in support for data I/O,
pre/post processing, and commonly used algorithms, 2) interactively
add/remove/reorder custom algorithms and adjust their parameters, and 3)
visualize results for classification, detection, segmentation, and tracking
tasks. Moreover, because it creates all the code files in structured
directories, it allows easy sharing of entire projects or even the individual
components to be reused by other researchers. The effectiveness of LiGuard is
demonstrated via case studies.

### 2. [PercepTwin: Modeling High-Fidelity Digital Twins for Sim2Real LiDAR-based Perception for Intelligent Transportation Systems](http://arxiv.org/pdf/2509.02903v1)

Authors: Muhammad Shahbaz, Shaurya Agarwal

LiDAR-based perception in intelligent transportation systems (ITS), for tasks
such as object detection, tracking, and semantic and instance segmentation, is
predominantly solved by deep neural network models which often require
large-scale labeled datasets during training to achieve generalization.
However, creating these datasets is costly. time consuming and require human
labor before the datasets are ready for training models. This hinders
scalability of the LiDAR-based perception systems in ITS. Sim2Real learning
offers scalable alternative, however, its effectiveness is dependent on the
fidelity of the source simulation(s) to real-world, in terms of environment
structure, actor dynamics, and sensor emulations. In response, this paper
introduces a rigorous and reproducible methodology for creating large-scale,
high-quality synthetic datasets using High-Fidelity Digital Twins (HiFi DTs).
The proposed workflow outlines the steps, tools, and best practices for
digitally replicating real-world environments, encompassing static geometry
modeling, road infrastructure replication, and dynamic traffic scenario
generation. Leveraging open-source and readily available resources such as
satellite imagery and OpenStreetMap data, alongside specific sensor
configurations, this paper provides practical, detailed guidance for
constructing robust synthetic environments. These environments subsequently
facilitate scalable, cost-effective, and diverse dataset generation, forming a
reliable foundation for robust Sim2Real learning.

### 3. [High-Fidelity Digital Twins for Bridging the Sim2Real Gap in LiDAR-Based ITS Perception](http://arxiv.org/pdf/2509.02904v1)

Authors: Muhammad Shahbaz, Shaurya Agarwal

Sim2Real domain transfer offers a cost-effective and scalable approach for
developing LiDAR-based perception (e.g., object detection, tracking,
segmentation) in Intelligent Transportation Systems (ITS). However, perception
models trained in simulation often under perform on real-world data due to
distributional shifts. To address this Sim2Real gap, this paper proposes a
high-fidelity digital twin (HiFi DT) framework that incorporates real-world
background geometry, lane-level road topology, and sensor-specific
specifications and placement. We formalize the domain adaptation challenge
underlying Sim2Real learning and present a systematic method for constructing
simulation environments that yield in-domain synthetic data. An off-the-shelf
3D object detector is trained on HiFi DT-generated synthetic data and evaluated
on real data. Our experiments show that the DT-trained model outperforms the
equivalent model trained on real data by 4.8%. To understand this gain, we
quantify distributional alignment between synthetic and real data using
multiple metrics, including Chamfer Distance (CD), Maximum Mean Discrepancy
(MMD), Earth Mover's Distance (EMD), and Fr'echet Distance (FD), at both
raw-input and latent-feature levels. Results demonstrate that HiFi DTs
substantially reduce domain shift and improve generalization across diverse
evaluation scenarios. These findings underscore the significant role of digital
twins in enabling reliable, simulation-based LiDAR perception for real-world
ITS applications.

### 4. [STAR: A Fast and Robust Rigid Registration Framework for Serial Histopathological Images](http://arxiv.org/pdf/2509.02952v1)

Authors: Zeyu Liu, Shengwei Ding

Registration of serial whole-slide histopathological images (WSIs) is
critical for enabling direct comparison across diverse stains and for preparing
paired datasets in artificial intelligence (AI) workflows such as virtual
staining and biomarker prediction. While existing methods often rely on complex
deformable or deep learning approaches that are computationally intensive and
difficult to reproduce, lightweight rigid frameworks-sufficient for many
consecutive-section scenarios-remain underdeveloped. We introduce STAR (Serial
Tissue Alignment for Rigid registration), a fast and robust open-source
framework for multi-WSI alignment. STAR integrates stain-conditioned
preprocessing with a hierarchical coarse-to-fine correlation strategy, adaptive
kernel scaling, and built-in quality control, achieving reliable rigid
registration across heterogeneous tissue types and staining protocols,
including hematoxylin-eosin (H&E), special histochemical stains (e.g., PAS,
PASM, Masson's), and immunohistochemical (IHC) markers (e.g., CD31, KI67).
Evaluated on the ANHIR 2019 and ACROBAT 2022 datasets spanning multiple organs
and scanning conditions, STAR consistently produced stable alignments within
minutes per slide, demonstrating robustness to cross-stain variability and
partial tissue overlap. Beyond benchmarks, we present case studies on H&E-IHC
alignment, construction of multi-IHC panels, and typical failure modes,
underscoring both utility and limitations. Released as an open and lightweight
tool, STAR provides a reproducible baseline that lowers the barrier for
clinical adoption and enables large-scale paired data preparation for
next-generation computational pathology.

### 5. [Resilient Multimodal Industrial Surface Defect Detection with Uncertain Sensors Availability](http://arxiv.org/pdf/2509.02962v1)

Authors: Shuai Jiang, Yunfeng Ma, Jingyu Zhou, Yuan Bian, Yaonan Wang, Min Liu

Multimodal industrial surface defect detection (MISDD) aims to identify and
locate defect in industrial products by fusing RGB and 3D modalities. This
article focuses on modality-missing problems caused by uncertain sensors
availability in MISDD. In this context, the fusion of multiple modalities
encounters several troubles, including learning mode transformation and
information vacancy. To this end, we first propose cross-modal prompt learning,
which includes: i) the cross-modal consistency prompt serves the establishment
of information consistency of dual visual modalities; ii) the modality-specific
prompt is inserted to adapt different input patterns; iii) the missing-aware
prompt is attached to compensate for the information vacancy caused by dynamic
modalities-missing. In addition, we propose symmetric contrastive learning,
which utilizes text modality as a bridge for fusion of dual vision modalities.
Specifically, a paired antithetical text prompt is designed to generate binary
text semantics, and triple-modal contrastive pre-training is offered to
accomplish multimodal learning. Experiment results show that our proposed
method achieves 73.83% I-AUROC and 93.05% P-AUROC with a total missing rate 0.7
for RGB and 3D modalities (exceeding state-of-the-art methods 3.84% and 5.58%
respectively), and outperforms existing approaches to varying degrees under
different missing types and rates. The source code will be available at
https://github.com/SvyJ/MISDD-MM.

### 6. [InstaDA: Augmenting Instance Segmentation Data with Dual-Agent System](http://arxiv.org/pdf/2509.02973v1)

Authors: Xianbao Hou, Yonghao He, Zeyd Boukhers, John See, Hu Su, Wei Sui, Cong Yang

Acquiring high-quality instance segmentation data is challenging due to the
labor-intensive nature of the annotation process and significant class
imbalances within datasets. Recent studies have utilized the integration of
Copy-Paste and diffusion models to create more diverse datasets. However, these
studies often lack deep collaboration between large language models (LLMs) and
diffusion models, and underutilize the rich information within the existing
training data. To address these limitations, we propose InstaDA, a novel,
training-free Dual-Agent system designed to augment instance segmentation
datasets. First, we introduce a Text-Agent (T-Agent) that enhances data
diversity through collaboration between LLMs and diffusion models. This agent
features a novel Prompt Rethink mechanism, which iteratively refines prompts
based on the generated images. This process not only fosters collaboration but
also increases image utilization and optimizes the prompts themselves.
Additionally, we present an Image-Agent (I-Agent) aimed at enriching the
overall data distribution. This agent augments the training set by generating
new instances conditioned on the training images. To ensure practicality and
efficiency, both agents operate as independent and automated workflows,
enhancing usability. Experiments conducted on the LVIS 1.0 validation set
indicate that InstaDA achieves significant improvements, with an increase of
+4.0 in box average precision (AP) and +3.3 in mask AP compared to the
baseline. Furthermore, it outperforms the leading model, DiverGen, by +0.3 in
box AP and +0.1 in mask AP, with a notable +0.7 gain in box AP on common
categories and mask AP gains of +0.2 on common categories and +0.5 on frequent
categories.

### 7. [SPENet: Self-guided Prototype Enhancement Network for Few-shot Medical Image Segmentation](http://arxiv.org/pdf/2509.02993v1)

Authors: Chao Fan, Xibin Jia, Anqi Xiao, Hongyuan Yu, Zhenghan Yang, Dawei Yang, Hui Xu, Yan Huang, Liang Wang

Few-Shot Medical Image Segmentation (FSMIS) aims to segment novel classes of
medical objects using only a few labeled images. Prototype-based methods have
made significant progress in addressing FSMIS. However, they typically generate
a single global prototype for the support image to match with the query image,
overlooking intra-class variations. To address this issue, we propose a
Self-guided Prototype Enhancement Network (SPENet). Specifically, we introduce
a Multi-level Prototype Generation (MPG) module, which enables
multi-granularity measurement between the support and query images by
simultaneously generating a global prototype and an adaptive number of local
prototypes. Additionally, we observe that not all local prototypes in the
support image are beneficial for matching, especially when there are
substantial discrepancies between the support and query images. To alleviate
this issue, we propose a Query-guided Local Prototype Enhancement (QLPE)
module, which adaptively refines support prototypes by incorporating guidance
from the query image, thus mitigating the negative effects of such
discrepancies. Extensive experiments on three public medical datasets
demonstrate that SPENet outperforms existing state-of-the-art methods,
achieving superior performance.

### 8. [SOPSeg: Prompt-based Small Object Instance Segmentation in Remote Sensing Imagery](http://arxiv.org/pdf/2509.03002v1)

Authors: Chenhao Wang, Yingrui Ji, Yu Meng, Yunjian Zhang, Yao Zhu

Extracting small objects from remote sensing imagery plays a vital role in
various applications, including urban planning, environmental monitoring, and
disaster management. While current research primarily focuses on small object
detection, instance segmentation for small objects remains underexplored, with
no dedicated datasets available. This gap stems from the technical challenges
and high costs of pixel-level annotation for small objects. While the Segment
Anything Model (SAM) demonstrates impressive zero-shot generalization, its
performance on small-object segmentation deteriorates significantly, largely
due to the coarse 1/16 feature resolution that causes severe loss of fine
spatial details. To this end, we propose SOPSeg, a prompt-based framework
specifically designed for small object segmentation in remote sensing imagery.
It incorporates a region-adaptive magnification strategy to preserve
fine-grained details, and employs a customized decoder that integrates edge
prediction and progressive refinement for accurate boundary delineation.
Moreover, we introduce a novel prompting mechanism tailored to the oriented
bounding boxes widely adopted in remote sensing applications. SOPSeg
outperforms existing methods in small object segmentation and facilitates
efficient dataset construction for remote sensing tasks. We further construct a
comprehensive small object instance segmentation dataset based on SODA-A, and
will release both the model and dataset to support future research.

### 9. [Enhancing Robustness in Post-Processing Watermarking: An Ensemble Attack Network Using CNNs and Transformers](http://arxiv.org/pdf/2509.03006v1)

Authors: Tzuhsuan Huang, Cheng Yu Yeo, Tsai-Ling Huang, Hong-Han Shuai, Wen-Huang Cheng, Jun-Cheng Chen

Recent studies on deep watermarking have predominantly focused on
in-processing watermarking, which integrates the watermarking process into
image generation. However, post-processing watermarking, which embeds
watermarks after image generation, offers more flexibility. It can be applied
to outputs from any generative model (e.g. GANs, diffusion models) without
needing access to the model's internal structure. It also allows users to embed
unique watermarks into individual images. Therefore, this study focuses on
post-processing watermarking and enhances its robustness by incorporating an
ensemble attack network during training. We construct various versions of
attack networks using CNN and Transformer in both spatial and frequency domains
to investigate how each combination influences the robustness of the
watermarking model. Our results demonstrate that combining a CNN-based attack
network in the spatial domain with a Transformer-based attack network in the
frequency domain yields the highest robustness in watermarking models.
Extensive evaluation on the WAVES benchmark, using average bit accuracy as the
metric, demonstrates that our ensemble attack network significantly enhances
the robustness of baseline watermarking methods under various stress tests. In
particular, for the Regeneration Attack defined in WAVES, our method improves
StegaStamp by 18.743%. The code is released
at:https://github.com/aiiu-lab/DeepRobustWatermark.

### 10. [Background Matters Too: A Language-Enhanced Adversarial Framework for Person Re-Identification](http://arxiv.org/pdf/2509.03032v1)

Authors: Kaicong Huang, Talha Azfar, Jack M. Reilly, Thomas Guggisberg, Ruimin Ke

Person re-identification faces two core challenges: precisely locating the
foreground target while suppressing background noise and extracting
fine-grained features from the target region. Numerous visual-only approaches
address these issues by partitioning an image and applying attention modules,
yet they rely on costly manual annotations and struggle with complex
occlusions. Recent multimodal methods, motivated by CLIP, introduce semantic
cues to guide visual understanding. However, they focus solely on foreground
information, but overlook the potential value of background cues. Inspired by
human perception, we argue that background semantics are as important as the
foreground semantics in ReID, as humans tend to eliminate background
distractions while focusing on target appearance. Therefore, this paper
proposes an end-to-end framework that jointly models foreground and background
information within a dual-branch cross-modal feature extraction pipeline. To
help the network distinguish between the two domains, we propose an
intra-semantic alignment and inter-semantic adversarial learning strategy.
Specifically, we align visual and textual features that share the same
semantics across domains, while simultaneously penalizing similarity between
foreground and background features to enhance the network's discriminative
power. This strategy drives the model to actively suppress noisy background
regions and enhance attention toward identity-relevant foreground cues.
Comprehensive experiments on two holistic and two occluded ReID benchmarks
demonstrate the effectiveness and generality of the proposed method, with
results that match or surpass those of current state-of-the-art approaches.

### Computers and Society

### 1. [AI-Generated Images for representing Individuals: Navigating the Thin Line Between Care and Bias](http://arxiv.org/pdf/2509.03071v1)

Authors: Julia C. Ahrend, Björn Döge, Tom M Duscher, Dario Rodighiero

This research discusses the figurative tensions that arise when using
portraits to represent individuals behind a dataset. In the broader effort to
communicate European data related to depression, the Kiel Science Communication
Network (KielSCN) team attempted to engage a wider audience by combining
interactive data graphics with AI-generated images of people. This article
examines the project's decisions and results, reflecting on the reaction from
the audience when information design incorporates figurative representations of
individuals within the data.

### 2. [Plan More, Debug Less: Applying Metacognitive Theory to AI-Assisted Programming Education](http://arxiv.org/pdf/2509.03171v1)

Authors: Tung Phung, Heeryung Choi, Mengyan Wu, Adish Singla, Christopher Brooks

The growing adoption of generative AI in education highlights the need to
integrate established pedagogical principles into AI-assisted learning
environments. This study investigates the potential of metacognitive theory to
inform AI-assisted programming education through a hint system designed around
the metacognitive phases of planning, monitoring, and evaluation. Upon request,
the system can provide three types of AI-generated hints--planning, debugging,
and optimization--to guide students at different stages of problem-solving.
Through a study with 102 students in an introductory data science programming
course, we find that students perceive and engage with planning hints most
highly, whereas optimization hints are rarely requested. We observe a
consistent association between requesting planning hints and achieving higher
grades across question difficulty and student competency. However, when facing
harder tasks, students seek additional debugging but not more planning support.
These insights contribute to the growing field of AI-assisted programming
education by providing empirical evidence on the importance of pedagogical
principles in AI-assisted learning.

### 3. [Bridging Gaps Between Student and Expert Evaluations of AI-Generated Programming Hints](http://arxiv.org/pdf/2509.03269v1)

Authors: Tung Phung, Mengyan Wu, Heeryung Choi, Gustavo Soares, Sumit Gulwani, Adish Singla, Christopher Brooks

Generative AI has the potential to enhance education by providing
personalized feedback to students at scale. Recent work has proposed techniques
to improve AI-generated programming hints and has evaluated their performance
based on expert-designed rubrics or student ratings. However, it remains
unclear how the rubrics used to design these techniques align with students'
perceived helpfulness of hints. In this paper, we systematically study the
mismatches in perceived hint quality from students' and experts' perspectives
based on the deployment of AI-generated hints in a Python programming course.
We analyze scenarios with discrepancies between student and expert evaluations,
in particular, where experts rated a hint as high-quality while the student
found it unhelpful. We identify key reasons for these discrepancies and
classify them into categories, such as hints not accounting for the student's
main concern or not considering previous help requests. Finally, we propose and
discuss preliminary results on potential methods to bridge these gaps, first by
extending the expert-designed quality rubric and then by adapting the hint
generation process, e.g., incorporating the student's comments or history.
These efforts contribute toward scalable, personalized, and pedagogically sound
AI-assisted feedback systems, which are particularly important for
high-enrollment educational settings.

### 4. [Integrating Generative AI into Cybersecurity Education: A Study of OCR and Multimodal LLM-assisted Instruction](http://arxiv.org/pdf/2509.02998v1)

Authors: Karan Patel, Yu-Zheng Lin, Gaurangi Raul, Bono Po-Jen Shih, Matthew W. Redondo, Banafsheh Saber Latibari, Jesus Pacheco, Soheil Salehi, Pratik Satam

This full paper describes an LLM-assisted instruction integrated with a
virtual cybersecurity lab platform. The digital transformation of Fourth
Industrial Revolution (4IR) systems is reshaping workforce needs, widening
skill gaps, especially among older workers. With rising emphasis on robotics,
automation, AI, and security, re-skilling and up-skilling are essential.
Generative AI can help build this workforce by acting as an instructional
assistant to support skill acquisition during experiential learning. We present
a generative AI instructional assistant integrated into a prior experiential
learning platform. The assistant employs a zero-shot OCR-LLM pipeline within
the legacy Cybersecurity Labs-as-a-Service (CLaaS) platform (2015). Text is
extracted from slide images using Tesseract OCR, then simplified instructions
are generated via a general-purpose LLM, enabling real-time instructional
support with minimal infrastructure. The system was evaluated in a live
university course where student feedback (n=42) averaged 7.83/10, indicating
strong perceived usefulness. A comparative study with multimodal LLMs that
directly interpret slide images showed higher performance on visually dense
slides, but the OCR-LLM pipeline provided comparable pedagogical value on
text-centric slides with much lower computational overhead and cost. This work
demonstrates that a lightweight, easily integrable pipeline can effectively
extend legacy platforms with modern generative AI, offering scalable
enhancements for student comprehension in technical education.

### 5. [SESGO: Spanish Evaluation of Stereotypical Generative Outputs](http://arxiv.org/pdf/2509.03329v1)

Authors: Melissa Robles, Catalina Bernal, Denniss Raigoso, Mateo Dulce Rubio

This paper addresses the critical gap in evaluating bias in multilingual
Large Language Models (LLMs), with a specific focus on Spanish language within
culturally-aware Latin American contexts. Despite widespread global deployment,
current evaluations remain predominantly US-English-centric, leaving potential
harms in other linguistic and cultural contexts largely underexamined. We
introduce a novel, culturally-grounded framework for detecting social biases in
instruction-tuned LLMs. Our approach adapts the underspecified question
methodology from the BBQ dataset by incorporating culturally-specific
expressions and sayings that encode regional stereotypes across four social
categories: gender, race, socioeconomic class, and national origin. Using more
than 4,000 prompts, we propose a new metric that combines accuracy with the
direction of error to effectively balance model performance and bias alignment
in both ambiguous and disambiguated contexts. To our knowledge, our work
presents the first systematic evaluation examining how leading commercial LLMs
respond to culturally specific bias in the Spanish language, revealing varying
patterns of bias manifestation across state-of-the-art models. We also
contribute evidence that bias mitigation techniques optimized for English do
not effectively transfer to Spanish tasks, and that bias patterns remain
largely consistent across different sampling temperatures. Our modular
framework offers a natural extension to new stereotypes, bias categories, or
languages and cultural contexts, representing a significant step toward more
equitable and culturally-aware evaluation of AI systems in the diverse
linguistic environments where they operate.

### 6. [More Parameters Than Populations: A Systematic Literature Review of Large Language Models within Survey Research](http://arxiv.org/pdf/2509.03391v1)

Authors: Trent D. Buskirk, Florian Keusch, Leah von der Heyde, Adam Eck

Survey research has a long-standing history of being a human-powered field,
but one that embraces various technologies for the collection, processing, and
analysis of various behavioral, political, and social outcomes of interest,
among others. At the same time, Large Language Models (LLMs) bring new
technological challenges and prerequisites in order to fully harness their
potential. In this paper, we report work-in-progress on a systematic literature
review based on keyword searches from multiple large-scale databases as well as
citation networks that assesses how LLMs are currently being applied within the
survey research process. We synthesize and organize our findings according to
the survey research process to include examples of LLM usage across three broad
phases: pre-data collection, data collection, and post-data collection. We
discuss selected examples of potential use cases for LLMs as well as its
pitfalls based on examples from existing literature. Considering survey
research has rich experience and history regarding data quality, we discuss
some opportunities and describe future outlooks for survey research to
contribute to the continued development and refinement of LLMs.

### 7. [The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices](http://arxiv.org/pdf/2509.02910v1)

Authors: Sandra C. Matz, C. Blaine Horton, Sofie Goethals

Large language models (LLMs) increasingly act on people's behalf: they write
emails, buy groceries, and book restaurants. While the outsourcing of human
decision-making to AI can be both efficient and effective, it raises a
fundamental question: how does delegating identity-defining choices to AI
reshape who people become? We study the impact of agentic LLMs on two
identity-relevant outcomes: interpersonal distinctiveness - how unique a
person's choices are relative to others - and intrapersonal diversity - the
breadth of a single person's choices over time. Using real choices drawn from
social-media behavior of 1,000 U.S. users (110,000 choices in total), we
compare a generic and personalized agent to a human baseline. Both agents shift
people's choices toward more popular options, reducing the distinctiveness of
their behaviors and preferences. While the use of personalized agents tempers
this homogenization (compared to the generic AI), it also more strongly
compresses the diversity of people's preference portfolios by narrowing what
they explore across topics and psychological affinities. Understanding how AI
agents might flatten human experience, and how using generic versus
personalized agents involves distinctiveness-diversity trade-offs, is critical
for designing systems that augment rather than constrain human agency, and for
safeguarding diversity in thought, taste, and expression.

### 8. [Event Detection and Classification for Long Range Sensing of Elephants Using Seismic Signal](http://arxiv.org/pdf/2509.02920v1)

Authors: Jaliya L. Wijayaraja, Janaka L. Wijekoon, Malitha Wijesundara

Detecting elephants through seismic signals is an emerging research topic
aimed at developing solutions for Human-Elephant Conflict (HEC). Despite the
promising results, such solutions heavily rely on manual classification of
elephant footfalls, which limits their applicability for real-time
classification in natural settings. To address this limitation and build on our
previous work, this study introduces a classification framework targeting
resource-constrained implementations, prioritizing both accuracy and
computational efficiency. As part of this framework, a novel event detection
technique named Contextually Customized Windowing (CCW), tailored specifically
for detecting elephant footfalls, was introduced, and evaluations were
conducted by comparing it with the Short-Term Average/Long-Term Average
(STA/LTA) method. The yielded results show that the maximum validated detection
range was 155.6 m in controlled conditions and 140 m in natural environments.
Elephant footfall classification using Support Vector Machine (SVM) with a
Radial Basis Function (RBF) kernel demonstrated superior performance across
multiple settings, achieving an accuracy of 99% in controlled environments, 73%
in natural elephant habitats, and 70% in HEC-prone human habitats, the most
challenging scenario. Furthermore, feature impact analysis using explainable AI
identified the number of Zero Crossings and Dynamic Time Warping (DTW)
Alignment Cost as the most influential factors in all experiments, while
Predominant Frequency exhibited significant influence in controlled settings.

### Databases

### 1. [CARPO: Leveraging Listwise Learning-to-Rank for Context-Aware Query Plan Optimization](http://arxiv.org/pdf/2509.03102v1)

Authors: Wenrui Zhou, Qiyu Liu, Jingshu Peng, Aoqian Zhang, Lei Chen

Efficient data processing is increasingly vital, with query optimizers
playing a fundamental role in translating SQL queries into optimal execution
plans. Traditional cost-based optimizers, however, often generate suboptimal
plans due to flawed heuristics and inaccurate cost models, leading to the
emergence of Learned Query Optimizers (LQOs). To address challenges in existing
LQOs, such as the inconsistency and suboptimality inherent in pairwise ranking
methods, we introduce CARPO, a generic framework leveraging listwise
learning-to-rank for context-aware query plan optimization. CARPO distinctively
employs a Transformer-based model for holistic evaluation of candidate plan
sets and integrates a robust hybrid decision mechanism, featuring
Out-Of-Distribution (OOD) detection with a top-$k$ fallback strategy to ensure
reliability. Furthermore, CARPO can be seamlessly integrated with existing plan
embedding techniques, demonstrating strong adaptability. Comprehensive
experiments on TPC-H and STATS benchmarks demonstrate that CARPO significantly
outperforms both native PostgreSQL and Lero, achieving a Top-1 Rate of
\textbf{74.54\%} on the TPC-H benchmark compared to Lero's 3.63\%, and reducing
the total execution time to 3719.16 ms compared to PostgreSQL's 22577.87 ms.

### 2. [BAMG: A Block-Aware Monotonic Graph Index for Disk-Based Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2509.03226v1)

Authors: Huiling Li, Jianliang Xu

Approximate Nearest Neighbor Search (ANNS) over high-dimensional vectors is a
foundational problem in databases, where disk I/O often emerges as the dominant
performance bottleneck at scale. Existing graph indexing solutions for
disk-based ANNS typically either optimize the storage layout for a given graph
or construct the graph independently of the storage layout, thus overlooking
their interaction. In this paper, we propose the Block-aware Monotonic Relative
Neighborhood Graph (BMRNG), a novel graph structure that jointly considers both
geometric distance and storage layout for edge selection, theoretically
guaranteeing the existence of I/O monotonic search paths. To address the
scalability challenge of BMRNG construction, we further develop a practical and
efficient variant, the Block-Aware Monotonic Graph (BAMG), which can be
constructed in linear time from a monotonic graph considering the storage
layout. BAMG integrates block-aware edge pruning with a decoupled storage
design that separates raw vectors from the graph index, thereby maximizing
block utilization and minimizing redundant disk reads. Additionally, we design
a multi-layer navigation graph for adaptive and efficient query entry, along
with a block-first search algorithm that prioritizes intra-block traversal to
fully exploit each disk I/O operation. Extensive experiments on real-world
datasets demonstrate that BAMG achieves up to 2.1x higher throughput and
reduces I/O reads by up to 52% compared to state-of-the-art methods, while
maintaining comparable recall.

### 3. [Adaptive KV-Cache Compression without Manually Setting Budget](http://arxiv.org/pdf/2509.03136v1)

Authors: Chenxia Tang, Jianchun Liu, Hongli Xu, Liusheng Huang

Large language models (LLMs) inference relies heavily on KV-caches to
accelerate autoregressive decoding, but the resulting memory footprint grows
rapidly with sequence length, posing significant efficiency challenges. Current
KV-cache compression methods suffer from a Procrustes' bed problem: they force
diverse workloads into fixed compression ratios, leading to suboptimal resource
allocation and inference performance. To this end, we present GVote, an
adaptive KV-cache compression scheme that eliminates manual budget
specification while achieving superior accuracy-efficiency trade-offs. GVote
operates on the principle that the important keys are the aggregation of keys
required by future queries. The method predicts future query attention demands
by Monte-Carlo style sampling potential queries and aggregating selected keys
to determine the optimal cache budget without manual specification.
Experimental evaluation demonstrates GVote's effectiveness across multiple
benchmarks, including GSM8K, RULER and Longbench. Compared to baselines, GVote
exhibits 2$\times$ memory reduction while the accuracy maintains higher or
comparable.

### 4. [NeurStore: Efficient In-database Deep Learning Model Management System](http://arxiv.org/pdf/2509.03228v1)

Authors: Siqi Xiang, Sheng Wang, Xiaokui Xiao, Cong Yue, Zhanhao Zhao, Beng Chin Ooi

With the prevalence of in-database AI-powered analytics, there is an
increasing demand for database systems to efficiently manage the ever-expanding
number and size of deep learning models. However, existing database systems
typically store entire models as monolithic files or apply compression
techniques that overlook the structural characteristics of deep learning
models, resulting in suboptimal model storage overhead. This paper presents
NeurStore, a novel in-database model management system that enables efficient
storage and utilization of deep learning models. First, NeurStore employs a
tensor-based model storage engine to enable fine-grained model storage within
databases. In particular, we enhance the hierarchical navigable small world
(HNSW) graph to index tensors, and only store additional deltas for tensors
within a predefined similarity threshold to ensure tensor-level deduplication.
Second, we propose a delta quantization algorithm that effectively compresses
delta tensors, thus achieving a superior compression ratio with controllable
model accuracy loss. Finally, we devise a compression-aware model loading
mechanism, which improves model utilization performance by enabling direct
computation on compressed tensors. Experimental evaluations demonstrate that
NeurStore achieves superior compression ratios and competitive model loading
throughput compared to state-of-the-art approaches.

### Distributed, Parallel, and Cluster Computing

### 1. [The High Cost of Keeping Warm: Characterizing Overhead in Serverless Autoscaling Policies](http://arxiv.org/pdf/2509.03104v1)

Authors: Leonid Kondrashov, Boxi Zhou, Hancheng Wang, Dmitrii Ustiugov

Serverless computing is transforming cloud application development, but the
performance-cost trade-offs of control plane designs remain poorly understood
due to a lack of open, cross-platform benchmarks and detailed system analyses.
In this work, we address these gaps by designing a serverless system that
approximates the scaling behaviors of commercial providers, including AWS
Lambda and Google Cloud Run. We systematically compare the performance and
cost-efficiency of both synchronous and asynchronous autoscaling policies by
replaying real-world workloads and varying key autoscaling parameters.
  We demonstrate that our open-source systems can closely replicate the
operational characteristics of commercial platforms, enabling reproducible and
transparent experimentation. By evaluating how autoscaling parameters affect
latency, memory usage, and CPU overhead, we reveal several key findings. First,
we find that serverless systems exhibit significant computational overhead due
to instance churn equivalent to 10-40% of the CPU cycles spent on request
handling, primarily originating from worker nodes. Second, we observe high
memory allocation due to scaling policy: 2-10 times more than actively used.
Finally, we demonstrate that reducing these overheads typically results in
significant performance degradation in the current systems, underscoring the
need for new, cost-efficient autoscaling strategies. Additionally, we employ a
hybrid methodology that combines real control plane deployments with
large-scale simulation to extend our evaluation closer to a production scale,
thereby bridging the gap between small research clusters and real-world
environments.

### 2. [Efficient and Secure Sleepy Model for BFT Consensus](http://arxiv.org/pdf/2509.03145v1)

Authors: Pengkun Ren, Hai Dong, Zahir Tari, Pengcheng Zhang

Byzantine Fault Tolerant (BFT) consensus protocols for dynamically available
systems face a critical challenge: balancing latency and security in
fluctuating node participation. Existing solutions often require multiple
rounds of voting per decision, leading to high latency or limited resilience to
adversarial behavior. This paper presents a BFT protocol integrating a
pre-commit mechanism with publicly verifiable secret sharing (PVSS) into
message transmission. By binding users' identities to their messages through
PVSS, our approach reduces communication rounds. Compared to other
state-of-the-art methods, our protocol typically requires only four network
delays (4$\Delta$) in common scenarios while being resilient to up to 1/2
adversarial participants. This integration enhances the efficiency and security
of the protocol without compromising integrity. Theoretical analysis
demonstrates the robustness of the protocol against Byzantine attacks.
Experimental evaluations show that, compared to traditional BFT protocols, our
protocol significantly prevents fork occurrences and improves chain stability.
Furthermore, compared to longest-chain protocol, our protocol maintains
stability and lower latency in scenarios with moderate participation
fluctuations.

### 3. [Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](http://arxiv.org/pdf/2509.03018v1)

Authors: Yangtao Deng, Lei Zhang, Qinlong Wang, Xiaoyun Zhi, Xinlei Zhang, Zhuo Jiang, Haohan Xu, Lei Wang, Zuquan Song, Gaohong Liu, Yang Bai, Shuguang Wang, Wencong Xiao, Jianxi Ye, Minlan Yu, Hong Xu

Reliability is essential for ensuring efficiency in LLM training. However,
many real-world reliability issues remain difficult to resolve, resulting in
wasted resources and degraded model performance. Unfortunately, today's
collective communication libraries operate as black boxes, hiding critical
information needed for effective root cause analysis. We propose Mycroft, a
lightweight distributed tracing and root cause analysis system designed to
address previously hidden reliability issues in collective communication.
Mycroft's key idea is to trace collective communication states and leverage
internal control and data dependencies to resolve reliability problems in LLM
training. Mycroft has been deployed at ByteDance for over six months to debug
collective communication related issues at runtime. It detected anomalies
within 15 seconds in 90% of cases and identified the root cause within 20
seconds in 60% of cases. We also conducted extensive fault injection
experiments to demonstrate Mycroft's capability and efficiency.

### 4. [FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs](http://arxiv.org/pdf/2509.03047v1)

Authors: Haijun Zhang, Jinxiang Wang, Zhenhua Yu, Yanyong Zhang, Xuejie Ji, Kaining Mao, Jun Zhang, Yaqing Zhang, Ting Wu, Fei Jie, Xiemin Huang, Zhifang Cai, Junhua Cheng, Shuwei Wang, Wei Li, Xiaoming Bao, Hua Xu, Shixiong Zhao, Jun Li, Hongwei Sun, Ziyang Zhang, Yi Xiong, Chunsheng Li

Large language models (LLMs) have made a profound impact across various
fields due to their advanced capabilities. However, training these models at
unprecedented scales requires extensive AI accelerator clusters and
sophisticated parallelism strategies, which pose significant challenges in
maintaining system reliability over prolonged training periods. A major concern
is the substantial loss of training time caused by inevitable hardware and
software failures. To address these challenges, we present FlashRecovery, a
fast and low-cost failure recovery system comprising three core modules: (1)
Active and real-time failure detection. This module performs continuous
training state monitoring, enabling immediate identification of hardware and
software failures within seconds, thus ensuring rapid incident response; (2)
Scale-independent task restart. By employing different recovery strategies for
normal and faulty nodes, combined with an optimized communication group
reconstruction protocol, our approach ensures that the recovery time remains
nearly constant, regardless of cluster scale; (3) Checkpoint-free recovery
within one step. Our novel recovery mechanism enables single-step restoration,
completely eliminating dependence on traditional checkpointing methods and
their associated overhead. Collectively, these innovations enable FlashRecovery
to achieve optimal Recovery Time Objective (RTO) and Recovery Point Objective
(RPO), substantially improving the reliability and efficiency of long-duration
LLM training. Experimental results demonstrate that FlashRecovery system can
achieve training restoration on training cluster with 4, 800 devices in 150
seconds. We also verify that the time required for failure recovery is nearly
consistent for different scales of training tasks.

### 5. [A description of the radio astronomy data processing tool DDF Pipeline](http://arxiv.org/pdf/2509.03075v1)

Authors: Mathis Certenais, François Bodin, Laurent Morin

This paper presents the DDF Pipeline, a radio astronomy data processing tool
initially designed for the LOw-Frequency ARray (LO- FAR) radio-telescope and a
candidate for processing data from the Square Kilometre Array (SKA). This work
describes the DDF Pipeline software and presents a coarse-grain profiling
execution to characterize its performance.

### 6. [Treasure Hunt in Anonymous Graphs with Quantum Pebbles by Oblivious Agents](http://arxiv.org/pdf/2509.02909v1)

Authors: Gaurav Gaur, Barun Gorain, Rishi Ranjan Singh, Daya Gaur

We investigate the problem of finding a static treasure in anonymous graphs
using oblivious agents and introduce a novel approach that leverages quantum
information. In anonymous graphs, vertices are unlabelled, indistinguishable,
and edges are locally labelled with port numbers. Agents typically rely on
stationary classical pebbles placed by an oracle to guide their search.
However, this classical approach is constrained by limited information
transmission and high traversal complexity. Classical pebbles are not
sufficient for search if the agents are oblivious. We propose the first use of
quantum pebbles for search in anonymous graphs. Quantum pebbles periodically
emit qubits in a fixed quantum state. Each pebble encodes the port number to
the next node using a unique quantum state. The agent determines the correct
path by performing measurements in multiple bases, exploiting the probabilistic
nature of quantum measurement to distinguish states. We show that this strategy
enables an oblivious agent to locate the treasure in $D$ steps using $D$
quantum pebbles, where $D$ is the length of the shortest path between the
starting point and the treasure. Moreover, only $O((\log D + \log \Delta)/(\log
1/\delta))$ measurements per node are required to ensure high success
probability in a graph with maximum degree $\Delta$ where $\delta =
\cos^2(\frac{\pi}{2\Delta})$. We propose the use of quantum information as a
guidance mechanism in anonymous graph search. We demonstrate that quantum
pebbles can not only emulate the functionality of classical pebbles but can do
so with improved efficiency, offering a promising direction for future
quantum-enhanced distributed algorithms.

### 7. [CloudFormer: An Attention-based Performance Prediction for Public Clouds with Unknown Workload](http://arxiv.org/pdf/2509.03394v1)

Authors: Amirhossein Shahbazinia, Darong Huang, Luis Costero, David Atienza

Cloud platforms are increasingly relied upon to host diverse,
resource-intensive workloads due to their scalability, flexibility, and
cost-efficiency. In multi-tenant cloud environments, virtual machines are
consolidated on shared physical servers to improve resource utilization. While
virtualization guarantees resource partitioning for CPU, memory, and storage,
it cannot ensure performance isolation. Competition for shared resources such
as last-level cache, memory bandwidth, and network interfaces often leads to
severe performance degradation. Existing management techniques, including VM
scheduling and resource provisioning, require accurate performance prediction
to mitigate interference. However, this remains challenging in public clouds
due to the black-box nature of VMs and the highly dynamic nature of workloads.
To address these limitations, we propose CloudFormer, a dual-branch
Transformer-based model designed to predict VM performance degradation in
black-box environments. CloudFormer jointly models temporal dynamics and
system-level interactions, leveraging 206 system metrics at one-second
resolution across both static and dynamic scenarios. This design enables the
model to capture transient interference effects and adapt to varying workload
conditions without scenario-specific tuning. Complementing the methodology, we
provide a fine-grained dataset that significantly expands the temporal
resolution and metric diversity compared to existing benchmarks. Experimental
results demonstrate that CloudFormer consistently outperforms state-of-the-art
baselines across multiple evaluation metrics, achieving robust generalization
across diverse and previously unseen workloads. Notably, CloudFormer attains a
mean absolute error (MAE) of just 7.8%, representing a substantial improvement
in predictive accuracy and outperforming existing methods at least by 28%.

### 8. [DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling](http://arxiv.org/pdf/2509.03472v1)

Authors: Yubo Gao, Renbo Tu, Gennady Pekhimenko, Nandita Vijaykumar

Differentially-Private SGD (DP-SGD) is a powerful technique to protect user
privacy when using sensitive data to train neural networks. During training,
converting model weights and activations into low-precision formats, i.e.,
quantization, can drastically reduce training times, energy consumption, and
cost, and is thus a widely used technique. In this work, we demonstrate that
quantization causes significantly higher accuracy degradation in DP-SGD
compared to regular SGD. We observe that this is caused by noise injection in
DP-SGD, which amplifies quantization variance, leading to disproportionately
large accuracy degradation. To address this challenge, we present QPQuant, a
dynamic quantization framework that adaptively selects a changing subset of
layers to quantize at each epoch. Our method combines two key ideas that
effectively reduce quantization variance: (i) probabilistic sampling of the
layers that rotates which layers are quantized every epoch, and (ii) loss-aware
layer prioritization, which uses a differentially private loss sensitivity
estimator to identify layers that can be quantized with minimal impact on model
quality. This estimator consumes a negligible fraction of the overall privacy
budget, preserving DP guarantees. Empirical evaluations on ResNet18, ResNet50,
and DenseNet121 across a range of datasets demonstrate that DPQuant
consistently outperforms static quantization baselines, achieving near
Pareto-optimal accuracy-compute trade-offs and up to 2.21x theoretical
throughput improvements on low-precision hardware, with less than 2% drop in
validation accuracy.

### Digital Libraries

### 1. [More Parameters Than Populations: A Systematic Literature Review of Large Language Models within Survey Research](http://arxiv.org/pdf/2509.03391v1)

Authors: Trent D. Buskirk, Florian Keusch, Leah von der Heyde, Adam Eck

Survey research has a long-standing history of being a human-powered field,
but one that embraces various technologies for the collection, processing, and
analysis of various behavioral, political, and social outcomes of interest,
among others. At the same time, Large Language Models (LLMs) bring new
technological challenges and prerequisites in order to fully harness their
potential. In this paper, we report work-in-progress on a systematic literature
review based on keyword searches from multiple large-scale databases as well as
citation networks that assesses how LLMs are currently being applied within the
survey research process. We synthesize and organize our findings according to
the survey research process to include examples of LLM usage across three broad
phases: pre-data collection, data collection, and post-data collection. We
discuss selected examples of potential use cases for LLMs as well as its
pitfalls based on examples from existing literature. Considering survey
research has rich experience and history regarding data quality, we discuss
some opportunities and describe future outlooks for survey research to
contribute to the continued development and refinement of LLMs.

### Discrete Mathematics

### 1. [Homotopy equivalence of digital pictures in $\mathbb{Z}^2$](http://arxiv.org/pdf/2509.03023v1)

Authors: Dae-Woong Lee, P. Christopher Staecker

We investigate the properties of digital homotopy in the context of digital
pictures $(X,\kappa,\bar \kappa)$, where $X\subsetneq \mathbb{Z}^n$ is a finite
set, $\kappa$ is an adjacency relation on $X$, and $\bar \kappa$ is an
adjacency relation on the complement of $X$. In particular we focus on homotopy
equivalence between digital pictures in $\mathbb{Z}^2$. We define a numerical
homotopy-type invariant for digital pictures in $\mathbb{Z}^2$ called the outer
perimeter, which is a basic tool for distinguishing homotopy types of digital
pictures. When a digital pictures has no holes, we show that it is homotopy
equivalent to its rc-convex hull, obtained by ``filling in the gaps'' of any
row or column. We show that a digital picture $(X,c_i,c_j)$ is homotopy
equivalent to only finitely many other digital pictures $(Y,c_i,c_j)$. At the
end of the paper, we raise a conjecture on the row-column-convex hull of a
digital picture.

### 2. [Fast approximation algorithms for the 1-median problem on real-world large graphs](http://arxiv.org/pdf/2509.03052v1)

Authors: Keisuke Ueta, Wei Wu, Mutsunori Yagiura

The 1-median problem (1MP) on undirected weighted graphs seeks to find a
facility location minimizing the total weighted distance to all customer nodes.
Although the 1MP can be solved exactly by computing the single-source shortest
paths from each customer node, such approaches become computationally expensive
on large-scale graphs with millions of nodes. In many real-world applications,
such as recommendation systems based on large-scale knowledge graphs, the
number of nodes (i.e., potential facility locations) is enormous, whereas the
number of customer nodes is relatively small and spatially concentrated. In
such cases, exhaustive graph exploration is not only inefficient but also
unnecessary. Leveraging this observation, we propose three approximation
algorithms that reduce computation by terminating Dijkstra's algorithm early.
We provide theoretical analysis showing that one of the proposed algorithms
guarantees an approximation ratio of 2, whereas the other two improve this
ratio to 1.618. We demonstrate that the lower bound of the approximation ratio
is 1.2 by presenting a specific instance. Moreover, we show that all proposed
algorithms return optimal solutions when the number of customer nodes is less
than or equal to three. Extensive experiments demonstrate that our algorithms
significantly outperform baseline exact methods in runtime while maintaining
near-optimal accuracy across all tested graph types. Notably, on grid graphs
with 10 million nodes, our algorithms obtains all optimal solutions within 1
millisecond, whereas the baseline exact method requires over 70 seconds on
average.

### 3. [Representation number of word-representable co-bipartite graph](http://arxiv.org/pdf/2509.03064v1)

Authors: Biswajit Das, Ramesh Hariharasubramanian

A graph $G = (V, E)$ is said to be word-representable if there exists a word
$w$ over the alphabet $V$ such that, for any two distinct letters $x, y \in V$,
the letters $x$ and $y$ alternate in $w$ if and only if $xy \in E$. A graph is
co-bipartite if its complement is bipartite. Therefore, the vertex set of a
co-bipartite graph can be partitioned into two disjoint subsets $X$ and $Y$
such that the subgraphs induced by $X$ and $Y$ are cliques.
  The concept of word-representability for graph classes has gained significant
attention in recent years. The book Words and Graphs by Sergey Kitaev and Vadim
Lozin presents examples of co-bipartite graphs that are not word-representable.
It is known that a graph is word-representable if and only if it admits a
semi-transitive orientation. Although the necessary and sufficient conditions
for the existence of a semi-transitive orientation in co-bipartite graphs have
been established, the characterization based on vertex ordering remains open.
In this paper, we present necessary and sufficient conditions for a
co-bipartite graph to be word-representable in terms of its vertex ordering.
Furthermore, based on this vertex ordering, we provide an algorithm to
construct a $3$-uniform word-representation for any word-representable
co-bipartite graph. Using this result, we prove that except for the permutation
graphs, the representation number of all other word-representable co-bipartite
graphs is $3$.

### 4. [Expansion of gap-planar graphs](http://arxiv.org/pdf/2509.03121v1)

Authors: David R. Wood

A graph is $k$-gap-planar if it has a drawing in the plane such that every
crossing can be charged to one of the two edges involved so that at most $k$
crossings are charged to each edge. We show this class of graphs has linear
expansion. In particular, every $r$-shallow minor of a $k$-gap-planar graph has
density $O(rk)$. Several extensions of this result are proved: for topological
minors, for $k$-cover-planar graphs, for $k$-gap-cover-planar graphs, and for
drawings on any surface. Application to graph colouring are presented.

### 5. [Line Graphs of Non-Word-Representable Graphs are Not Always Non-Word-Representable](http://arxiv.org/pdf/2509.03339v1)

Authors: Khyodeno Mozhui, Tithi Dwary, K. V. Krishna

A graph is said to be word-representable if there exists a word over its
vertex set such that any two vertices are adjacent if and only if they
alternate in the word. If no such word exists, the graph is
non-word-representable. In the literature, there are examples of
non-word-representable graphs whose line graphs are non-word-representable.
However, it is an open problem to determine whether the line graph of a
non-word-representable graph is always non-word-representable or not? In this
work, we address the open problem by considering a class of
non-word-representable graphs, viz., Mycielski graphs of odd cycles of length
at least five, and show that their line graphs are word-representable.

### 6. [Row Impartial Terminus](http://arxiv.org/pdf/2509.03390v1)

Authors: Eric Gottlieb, Dawood Khatana, Matjaž Krnc, Peter Muršič, Ismael Qureshi

We introduce Row Impartial Terminus (RIT), an impartial combinatorial game
played on integer partitions. We show that any position in RIT can be uniquely
decomposed into a core and a remnant. Our central result is that the Conway
pair of any RIT position-which determines the outcome under both normal and
mis\`ere play-is identical to the Conway pair of a corresponding position in
the game of Nim defined by the remnant. This finding provides a complete
winning strategy for both variants of RIT, reducing its analysis to the
well-understood framework of Nim. As a consequence, we classify RIT within the
Conway-Gurvich-Ho hierarchy, showing it to be forced and miserable but not pet.

### Data Structures and Algorithms

### 1. [Compressed Dictionary Matching on Run-Length Encoded Strings](http://arxiv.org/pdf/2509.03265v1)

Authors: Philip Bille, Inge Li Gørtz, Simon J. Puglisi, Simon R. Tarnow

Given a set of pattern strings $\mathcal{P}=\{P_1, P_2,\ldots P_k\}$ and a
text string $S$, the classic dictionary matching problem is to report all
occurrences of each pattern in $S$. We study the dictionary problem in the
compressed setting, where the pattern strings and the text string are
compressed using run-length encoding, and the goal is to solve the problem
without decompression and achieve efficient time and space in the size of the
compressed strings. Let $m$ and $n$ be the total length of the patterns
$\mathcal{P}$ and the length of the text string $S$, respectively, and let
$\overline{m}$ and $\overline{n}$ be the total number of runs in the run-length
encoding of the patterns in $\mathcal{P}$ and $S$, respectively. Our main
result is an algorithm that achieves $O( (\overline{m} + \overline{n})\log \log
m + \mathrm{occ})$ expected time, and $O(\overline{m})$ space, where
$\mathrm{occ}$ is the total number of occurrences of patterns in $S$. This is
the first non-trivial solution to the problem. Since any solution must read the
input, our time bound is optimal within an $\log \log m$ factor. We introduce
several new techniques to achieve our bounds, including a new compressed
representation of the classic Aho-Corasick automaton and a new efficient string
index that supports fast queries in run-length encoded strings.

### 2. [Fast approximation algorithms for the 1-median problem on real-world large graphs](http://arxiv.org/pdf/2509.03052v1)

Authors: Keisuke Ueta, Wei Wu, Mutsunori Yagiura

The 1-median problem (1MP) on undirected weighted graphs seeks to find a
facility location minimizing the total weighted distance to all customer nodes.
Although the 1MP can be solved exactly by computing the single-source shortest
paths from each customer node, such approaches become computationally expensive
on large-scale graphs with millions of nodes. In many real-world applications,
such as recommendation systems based on large-scale knowledge graphs, the
number of nodes (i.e., potential facility locations) is enormous, whereas the
number of customer nodes is relatively small and spatially concentrated. In
such cases, exhaustive graph exploration is not only inefficient but also
unnecessary. Leveraging this observation, we propose three approximation
algorithms that reduce computation by terminating Dijkstra's algorithm early.
We provide theoretical analysis showing that one of the proposed algorithms
guarantees an approximation ratio of 2, whereas the other two improve this
ratio to 1.618. We demonstrate that the lower bound of the approximation ratio
is 1.2 by presenting a specific instance. Moreover, we show that all proposed
algorithms return optimal solutions when the number of customer nodes is less
than or equal to three. Extensive experiments demonstrate that our algorithms
significantly outperform baseline exact methods in runtime while maintaining
near-optimal accuracy across all tested graph types. Notably, on grid graphs
with 10 million nodes, our algorithms obtains all optimal solutions within 1
millisecond, whereas the baseline exact method requires over 70 seconds on
average.

### 3. [Triangle Detection in Worst-Case Sparse Graphs via Local Sketching](http://arxiv.org/pdf/2509.03215v1)

Authors: Hongyi Duan, Jian'an Zhang

We present a non-algebraic, locality-preserving framework for triangle
detection in worst-case sparse graphs. Our algorithm processes the graph in
$O(\log n)$ independent layers and partitions incident edges into prefix-based
classes where each class maintains a 1-sparse triple over a prime field.
Potential witnesses are surfaced by pair-key (PK) alignment, and every
candidate is verified by a three-stage, zero-false-positive pipeline: a
class-level 1-sparse consistency check, two slot-level decodings, and a final
adjacency confirmation. \textbf{To obtain single-run high-probability coverage,
we further instantiate $R=c_G\log n$ independent PK groups per class (each
probing a constant number of complementary buckets), which amplifies the
per-layer hit rate from $\Theta(1/\log n)$ to $1-n^{-\Omega(1)}$ without
changing the accounting.} A one-shot pairing discipline and class-term
triggering yield a per-(layer,level) accounting bound of $O(m)$, while
keep-coin concentration ensures that each vertex retains only $O(d^+(x))$ keys
with high probability. Consequently, the total running time is $O(m\log^2 n)$
and the peak space is $O(m\log n)$, both with high probability. The algorithm
emits a succinct Seeds+Logs artifact that enables a third party to replay all
necessary checks and certify a NO-instance in $\tilde O(m\log n)$ time. We also
prove a $\Theta(1/\log n)$ hit-rate lower bound for any single PK family under
a constant-probe local model (via Yao)--motivating the use of $\Theta(\log n)$
independent groups--and discuss why global algebraic convolutions would break
near-linear accounting or run into fine-grained barriers. We outline measured
paths toward Las Vegas $O(m\log n)$ and deterministic near-linear variants.

### 4. [Treasure Hunt in Anonymous Graphs with Quantum Pebbles by Oblivious Agents](http://arxiv.org/pdf/2509.02909v1)

Authors: Gaurav Gaur, Barun Gorain, Rishi Ranjan Singh, Daya Gaur

We investigate the problem of finding a static treasure in anonymous graphs
using oblivious agents and introduce a novel approach that leverages quantum
information. In anonymous graphs, vertices are unlabelled, indistinguishable,
and edges are locally labelled with port numbers. Agents typically rely on
stationary classical pebbles placed by an oracle to guide their search.
However, this classical approach is constrained by limited information
transmission and high traversal complexity. Classical pebbles are not
sufficient for search if the agents are oblivious. We propose the first use of
quantum pebbles for search in anonymous graphs. Quantum pebbles periodically
emit qubits in a fixed quantum state. Each pebble encodes the port number to
the next node using a unique quantum state. The agent determines the correct
path by performing measurements in multiple bases, exploiting the probabilistic
nature of quantum measurement to distinguish states. We show that this strategy
enables an oblivious agent to locate the treasure in $D$ steps using $D$
quantum pebbles, where $D$ is the length of the shortest path between the
starting point and the treasure. Moreover, only $O((\log D + \log \Delta)/(\log
1/\delta))$ measurements per node are required to ensure high success
probability in a graph with maximum degree $\Delta$ where $\delta =
\cos^2(\frac{\pi}{2\Delta})$. We propose the use of quantum information as a
guidance mechanism in anonymous graph search. We demonstrate that quantum
pebbles can not only emulate the functionality of classical pebbles but can do
so with improved efficiency, offering a promising direction for future
quantum-enhanced distributed algorithms.

### Emerging Technologies

### 1. [Treasure Hunt in Anonymous Graphs with Quantum Pebbles by Oblivious Agents](http://arxiv.org/pdf/2509.02909v1)

Authors: Gaurav Gaur, Barun Gorain, Rishi Ranjan Singh, Daya Gaur

We investigate the problem of finding a static treasure in anonymous graphs
using oblivious agents and introduce a novel approach that leverages quantum
information. In anonymous graphs, vertices are unlabelled, indistinguishable,
and edges are locally labelled with port numbers. Agents typically rely on
stationary classical pebbles placed by an oracle to guide their search.
However, this classical approach is constrained by limited information
transmission and high traversal complexity. Classical pebbles are not
sufficient for search if the agents are oblivious. We propose the first use of
quantum pebbles for search in anonymous graphs. Quantum pebbles periodically
emit qubits in a fixed quantum state. Each pebble encodes the port number to
the next node using a unique quantum state. The agent determines the correct
path by performing measurements in multiple bases, exploiting the probabilistic
nature of quantum measurement to distinguish states. We show that this strategy
enables an oblivious agent to locate the treasure in $D$ steps using $D$
quantum pebbles, where $D$ is the length of the shortest path between the
starting point and the treasure. Moreover, only $O((\log D + \log \Delta)/(\log
1/\delta))$ measurements per node are required to ensure high success
probability in a graph with maximum degree $\Delta$ where $\delta =
\cos^2(\frac{\pi}{2\Delta})$. We propose the use of quantum information as a
guidance mechanism in anonymous graph search. We demonstrate that quantum
pebbles can not only emulate the functionality of classical pebbles but can do
so with improved efficiency, offering a promising direction for future
quantum-enhanced distributed algorithms.

### 2. [Programmable Quantum Matter: Heralding Large Cluster States in Driven Inhomogeneous Spin Ensembles](http://arxiv.org/pdf/2509.02992v1)

Authors: Pratyush Anand, Louis Follet, Odiel Hooybergs, Dirk R. Englund

Atom-like emitters in solids are promising platforms for quantum sensing and
information processing, but inhomogeneities in the emitter fine structure
complicate quantum control. We present a framework that leverages this
diversity to reduce the resources for generating optically heralded spin
cluster states across $N_q$ emitters from the conventional order $O(N_q)$ to
$O(1)$ in ensembles of $N_q \sim 10$-$100$. An optimized pulse sequence
simultaneously corrects pulse-length and detuning errors, achieving
single-qubit gate fidelities exceeding $99.99\%$ for errors (normalized
relative to the Rabi drive strength) up to 0.3, while maintaining fidelities
above $99\%$ for errors as large as 0.4. Applied as a Carr-Purcell-Meiboom-Gill
(CPMG) dynamical decoupling protocol to the dominant noise spectrum of
silicon-vacancy centers in diamond, it enhances ensemble coherence times by
over $7\times$ compared to interleaved bang-bang based CPMG. For
state-of-the-art dilution refrigerators, global resonant optimal decoupling
across $N_q$ spins sharply reduces heating, addressing the trade-off between
the spin coherence and scaling to $N_q \gg 1$. We further introduce a modified
single-photon entanglement protocol with an efficient algorithm for
deterministic entanglement compilation. Depending on the decoupling time
window, our method yields order $O(10^2$-$10^4)$ more entanglement links than
bang-bang sequences, with theoretical guarantees of order $\Omega(N_q)$ unique
links, improvable by control tuning. Together, these techniques provide
scalable tools - including global control, phase denoising, remote
entanglement, and compilation - for robust quantum computing architectures with
heterogeneous spin ensembles.

### 3. [TraceLLM: Security Diagnosis Through Traces and Smart Contracts in Ethereum](http://arxiv.org/pdf/2509.03037v1)

Authors: Shuzheng Wang, Yue Huang, Zhuoer Xu, Yuming Huang, Jing Tang

Ethereum smart contracts hold tens of billions of USD in DeFi and NFTs, yet
comprehensive security analysis remains difficult due to unverified code,
proxy-based architectures, and the reliance on manual inspection of complex
execution traces. Existing approaches fall into two main categories: anomaly
transaction detection, which flags suspicious transactions but offers limited
insight into specific attack strategies hidden in execution traces inside
transactions, and code vulnerability detection, which cannot analyze unverified
contracts and struggles to show how identified flaws are exploited in real
incidents. As a result, analysts must still manually align transaction traces
with contract code to reconstruct attack scenarios and conduct forensics. To
address this gap, TraceLLM is proposed as a framework that leverages LLMs to
integrate execution trace-level detection with decompiled contract code. We
introduce a new anomaly execution path identification algorithm and an
LLM-refined decompile tool to identify vulnerable functions and provide
explicit attack paths to LLM. TraceLLM establishes the first benchmark for
joint trace and contract code-driven security analysis. For comparison, proxy
baselines are created by jointly transmitting the results of three
representative code analysis along with raw traces to LLM. TraceLLM identifies
attacker and victim addresses with 85.19\% precision and produces automated
reports with 70.37\% factual precision across 27 cases with ground truth expert
reports, achieving 25.93\% higher accuracy than the best baseline. Moreover,
across 148 real-world Ethereum incidents, TraceLLM automatically generates
reports with 66.22\% expert-verified accuracy, demonstrating strong
generalizability.

### 4. [Event Detection and Classification for Long Range Sensing of Elephants Using Seismic Signal](http://arxiv.org/pdf/2509.02920v1)

Authors: Jaliya L. Wijayaraja, Janaka L. Wijekoon, Malitha Wijesundara

Detecting elephants through seismic signals is an emerging research topic
aimed at developing solutions for Human-Elephant Conflict (HEC). Despite the
promising results, such solutions heavily rely on manual classification of
elephant footfalls, which limits their applicability for real-time
classification in natural settings. To address this limitation and build on our
previous work, this study introduces a classification framework targeting
resource-constrained implementations, prioritizing both accuracy and
computational efficiency. As part of this framework, a novel event detection
technique named Contextually Customized Windowing (CCW), tailored specifically
for detecting elephant footfalls, was introduced, and evaluations were
conducted by comparing it with the Short-Term Average/Long-Term Average
(STA/LTA) method. The yielded results show that the maximum validated detection
range was 155.6 m in controlled conditions and 140 m in natural environments.
Elephant footfall classification using Support Vector Machine (SVM) with a
Radial Basis Function (RBF) kernel demonstrated superior performance across
multiple settings, achieving an accuracy of 99% in controlled environments, 73%
in natural elephant habitats, and 70% in HEC-prone human habitats, the most
challenging scenario. Furthermore, feature impact analysis using explainable AI
identified the number of Zero Crossings and Dynamic Time Warping (DTW)
Alignment Cost as the most influential factors in all experiments, while
Predominant Frequency exhibited significant influence in controlled settings.

### Formal Languages and Automata Theory

### 1. [Identifiability and minimality bounds of quantum and post-quantum models of classical stochastic processes](http://arxiv.org/pdf/2509.03004v1)

Authors: Paul M. Riechers, Thomas J. Elliott

To make sense of the world around us, we develop models, constructed to
enable us to replicate, describe, and explain the behaviours we see. Focusing
on the broad case of sequences of correlated random variables, i.e., classical
stochastic processes, we tackle the question of determining whether or not two
different models produce the same observable behavior. This is the problem of
identifiability. Curiously, the physics of the model need not correspond to the
physics of the observations; recent work has shown that it is even advantageous
-- in terms of memory and thermal efficiency -- to employ quantum models to
generate classical stochastic processes. We resolve the identifiability problem
in this regime, providing a means to compare any two models of a classical
process, be the models classical, quantum, or `post-quantum', by mapping them
to a canonical `generalized' hidden Markov model. Further, this enables us to
place (sometimes tight) bounds on the minimal dimension required of a quantum
model to generate a given classical stochastic process.

### Graphics

### 1. [EclipseTouch: Touch Segmentation on Ad Hoc Surfaces using Worn Infrared Shadow Casting](http://arxiv.org/pdf/2509.03430v1)

Authors: Vimal Mollyn, Nathan DeVrio, Chris Harrison

The ability to detect touch events on uninstrumented, everyday surfaces has
been a long-standing goal for mixed reality systems. Prior work has shown that
virtual interfaces bound to physical surfaces offer performance and ergonomic
benefits over tapping at interfaces floating in the air. A wide variety of
approaches have been previously developed, to which we contribute a new
headset-integrated technique called \systemname. We use a combination of a
computer-triggered camera and one or more infrared emitters to create
structured shadows, from which we can accurately estimate hover distance (mean
error of 6.9~mm) and touch contact (98.0\% accuracy). We discuss how our
technique works across a range of conditions, including surface material,
interaction orientation, and environmental lighting.

### 2. [SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data](http://arxiv.org/pdf/2509.03451v1)

Authors: Nathan DeVrio, Vimal Mollyn, Chris Harrison

The ability to track a user's arm pose could be valuable in a wide range of
applications, including fitness, rehabilitation, augmented reality input, life
logging, and context-aware assistants. Unfortunately, this capability is not
readily available to consumers. Systems either require cameras, which carry
privacy issues, or utilize multiple worn IMUs or markers. In this work, we
describe how an off-the-shelf smartphone and smartwatch can work together to
accurately estimate arm pose. Moving beyond prior work, we take advantage of
more recent ultra-wideband (UWB) functionality on these devices to capture
absolute distance between the two devices. This measurement is the perfect
complement to inertial data, which is relative and suffers from drift. We
quantify the performance of our software-only approach using off-the-shelf
devices, showing it can estimate the wrist and elbow joints with a \hl{median
positional error of 11.0~cm}, without the user having to provide training data.

### Computer Science and Game Theory

### 1. [Zero-Error Nash Equilibrium: Harnessing Nonlocal Correlation in Incomplete Information Games](http://arxiv.org/pdf/2509.02947v1)

Authors: Ambuj, Tushar, Siddharth R. Pandey, Ram Krishna Patra, Anandamay Das Bhowmik, Kuntal Som, Amit Mukherjee

Claude Shannon's zero-error communication paradigm reshaped our understanding
of fault-tolerant information transfer. Here, we adapt this notion into game
theory with incomplete information. We ask: can players with private
information coordinate on a Nash equilibrium with zero probability of error? We
identify Bayesian games in which such coordination is impossible classically,
yet achievable by harnessing Bell nonlocal correlations. We formalize this
requirement as zero-error Nash equilibrium coordination, establishing a new
bridge between information theory, game theory, and quantum nonlocality.
Furthermore, we construct a tripartite Bayesian game that admits zero-error
Nash equilibrium coordination with genuine entanglement, and a two-player game
where a stronger notion of coordination can be achieved using every two-qubit
pure entangled state except the maximally one. Crucially, the advantage
persists under experimentally relevant noise, demonstrating nonlocality as a
robust resource for near-zero error decision-making under uncertainty.

### 2. [Generative Auto-Bidding in Large-Scale Competitive Auctions via Diffusion Completer-Aligner](http://arxiv.org/pdf/2509.03348v1)

Authors: Yewen Li, Jingtong Gao, Nan Jiang, Shuai Mao, Ruyi An, Fei Pan, Xiangyu Zhao, Bo An, Qingpeng Cai, Peng Jiang

Auto-bidding is central to computational advertising, achieving notable
commercial success by optimizing advertisers' bids within economic constraints.
Recently, large generative models show potential to revolutionize auto-bidding
by generating bids that could flexibly adapt to complex, competitive
environments. Among them, diffusers stand out for their ability to address
sparse-reward challenges by focusing on trajectory-level accumulated rewards,
as well as their explainable capability, i.e., planning a future trajectory of
states and executing bids accordingly. However, diffusers struggle with
generation uncertainty, particularly regarding dynamic legitimacy between
adjacent states, which can lead to poor bids and further cause significant loss
of ad impression opportunities when competing with other advertisers in a
highly competitive auction environment. To address it, we propose a Causal
auto-Bidding method based on a Diffusion completer-aligner framework, termed
CBD. Firstly, we augment the diffusion training process with an extra random
variable t, where the model observes t-length historical sequences with the
goal of completing the remaining sequence, thereby enhancing the generated
sequences' dynamic legitimacy. Then, we employ a trajectory-level return model
to refine the generated trajectories, aligning more closely with advertisers'
objectives. Experimental results across diverse settings demonstrate that our
approach not only achieves superior performance on large-scale auto-bidding
benchmarks, such as a 29.9% improvement in conversion value in the challenging
sparse-reward auction setting, but also delivers significant improvements on
the Kuaishou online advertising platform, including a 2.0% increase in target
cost.

### Human-Computer Interaction

### 1. [Demonstrating Visual Information Manipulation Attacks in Augmented Reality: A Hands-On Miniature City-Based Setup](http://arxiv.org/pdf/2509.02933v1)

Authors: Yanming Xiu, Maria Gorlatova

Augmented reality (AR) enhances user interaction with the real world but also
presents vulnerabilities, particularly through Visual Information Manipulation
(VIM) attacks. These attacks alter important real-world visual cues, leading to
user confusion and misdirected actions. In this demo, we present a hands-on
experience using a miniature city setup, where users interact with manipulated
AR content via the Meta Quest 3. The demo highlights the impact of VIM attacks
on user decision-making and underscores the need for effective security
measures in AR systems. Future work includes a user study and cross-platform
testing.

### 2. [OPRA-Vis: Visual Analytics System to Assist Organization-Public Relationship Assessment with Large Language Models](http://arxiv.org/pdf/2509.03164v1)

Authors: Sangbong Yoo, Seongbum Seo, Chanyoung Yoon, Hyelim Lee, Jeong-Nam Kim, Chansoo Kim, Yun Jang, Takanori Fujiwara

Analysis of public opinions collected from digital media helps organizations
maintain positive relationships with the public. Such public relations (PR)
analysis often involves assessing opinions, for example, measuring how strongly
people trust an organization. Pre-trained Large Language Models (LLMs) hold
great promise for supporting Organization-Public Relationship Assessment (OPRA)
because they can map unstructured public text to OPRA dimensions and articulate
rationales through prompting. However, adapting LLMs for PR analysis typically
requires fine-tuning on large labeled datasets, which is both labor-intensive
and knowledge-intensive, making it difficult for PR researchers to apply these
models. In this paper, we present OPRA-Vis, a visual analytics system that
leverages LLMs for OPRA without requiring extensive labeled data. Our framework
employs Chain-of-Thought prompting to guide LLMs in analyzing public opinion
data by incorporating PR expertise directly into the reasoning process.
Furthermore, OPRA-Vis provides visualizations that reveal the clues and
reasoning paths used by LLMs, enabling users to explore, critique, and refine
model decisions. We demonstrate the effectiveness of OPRA-Vis through two
real-world use cases and evaluate it quantitatively, through comparisons with
alternative LLMs and prompting strategies, and qualitatively, through
assessments of usability, effectiveness, and expert feedback.

### 3. [Finding My Way: Influence of Different Audio Augmented Reality Navigation Cues on User Experience and Subjective Usefulness](http://arxiv.org/pdf/2509.03199v1)

Authors: Sina Hinzmann, Francesco Vona, Juliane Henning, Mohamed Amer, Omar Abdellatif, Tanja Kojic, Jan-Niklas Voigt-Antons

As augmented reality (AR) becomes increasingly prevalent in mobile and
context-aware applications, the role of auditory cues in guiding users through
physical environments is becoming critical. This study investigates the
effectiveness and user experience of various categories of audio cues,
including fully non-verbal sounds and speech-derived Spearcons, during outdoor
navigation tasks using the Meta Quest 3 headset. Twenty participants navigated
five outdoor routes using audio-only cue types: Artificial Sounds, Nature
Sounds, Spearcons, Musical Instruments, and Auditory Icons. Subjective
evaluations were collected to assess the perceived effectiveness and user
experience of each sound type. Results revealed significant differences in
perceived novelty and stimulation across sound types. Artificial Sounds and
Musical Instruments were rated higher than Spearcons in novelty, while
Artificial Sounds were also rated higher than Spearcons in stimulation. Overall
preference was evenly split between Nature Sounds and Artificial Sounds. These
findings suggest that incorporating aspects of novelty and user engagement in
auditory feedback design may enhance the effectiveness of AR navigation
systems.

### 4. [Card Sorting with Fewer Cards and the Same Mental Models? A Re-examination of an Established Practice](http://arxiv.org/pdf/2509.03232v1)

Authors: Eduard Kuric, Peter Demcak, Matus Krajcovic

To keep card sorting with a lot of cards concise, a common strategy for
gauging mental models involves presenting participants with fewer randomly
selected cards instead of the full set. This is a decades-old practice, but its
effects lacked systematic examination. To assess how randomized subsets affect
data, we conducted an experiment with 160 participants. We compared results
between full and randomized 60\% card sets, then analyzed sample size
requirements and the impacts of individual personality and cognitive factors.
Our results demonstrate that randomized subsets can yield comparable similarity
matrices to standard card sorting, but thematic patterns in categories can
differ. Increased data variability also warrants larger sample sizes (25-35 for
60% card subset). Results indicate that personality traits and cognitive
reflection interact with card sorting. Our research suggests evidence-based
practices for conducting card sorting while exposing the influence of study
design and individual differences on measurement of mental models.

### 5. [Beyond Quantification: Navigating Uncertainty in Professional AI Systems](http://arxiv.org/pdf/2509.03271v1)

Authors: Sylvie Delacroix, Diana Robinson, Umang Bhatt, Jacopo Domenicucci, Jessica Montgomery, Gael Varoquaux, Carl Henrik Ek, Vincent Fortuin, Yulan He, Tom Diethe, Neill Campbell, Mennatallah El-Assady, Soren Hauberg, Ivana Dusparic, Neil Lawrence

The growing integration of large language models across professional domains
transforms how experts make critical decisions in healthcare, education, and
law. While significant research effort focuses on getting these systems to
communicate their outputs with probabilistic measures of reliability, many
consequential forms of uncertainty in professional contexts resist such
quantification. A physician pondering the appropriateness of documenting
possible domestic abuse, a teacher assessing cultural sensitivity, or a
mathematician distinguishing procedural from conceptual understanding face
forms of uncertainty that cannot be reduced to percentages. This paper argues
for moving beyond simple quantification toward richer expressions of
uncertainty essential for beneficial AI integration. We propose participatory
refinement processes through which professional communities collectively shape
how different forms of uncertainty are communicated. Our approach acknowledges
that uncertainty expression is a form of professional sense-making that
requires collective development rather than algorithmic optimization.

### 6. [More AI Assistance Reduces Cognitive Engagement: Examining the AI Assistance Dilemma in AI-Supported Note-Taking](http://arxiv.org/pdf/2509.03392v1)

Authors: Xinyue Chen, Kunlin Ruan, Kexin Phyllis Ju, Nathan Yap, Xu Wang

As AI tools become increasingly embedded in cognitively demanding tasks such
as note-taking, questions remain about whether they enhance or undermine
cognitive engagement. This paper examines the "AI Assistance Dilemma" in
note-taking, investigating how varying levels of AI support affect user
engagement and comprehension. In a within-subject experiment, we asked
participants (N=30) to take notes during lecture videos under three conditions:
Automated AI (high assistance with structured notes), Intermediate AI (moderate
assistance with real-time summary, and Minimal AI (low assistance with
transcript). Results reveal that Intermediate AI yields the highest post-test
scores and Automated AI the lowest. Participants, however, preferred the
automated setup due to its perceived ease of use and lower cognitive effort,
suggesting a discrepancy between preferred convenience and cognitive benefits.
Our study provides insights into designing AI assistance that preserves
cognitive engagement, offering implications for designing moderate AI support
in cognitive tasks.

### 7. [Beyond Words: Interjection Classification for Improved Human-Computer Interaction](http://arxiv.org/pdf/2509.03181v1)

Authors: Yaniv Goren, Yuval Cohen, Alexander Apartsin, Yehudit Aperstein

In the realm of human-computer interaction, fostering a natural dialogue
between humans and machines is paramount. A key, often overlooked, component of
this dialogue is the use of interjections such as "mmm" and "hmm". Despite
their frequent use to express agreement, hesitation, or requests for
information, these interjections are typically dismissed as "non-words" by
Automatic Speech Recognition (ASR) engines. Addressing this gap, we introduce a
novel task dedicated to interjection classification, a pioneer in the field to
our knowledge. This task is challenging due to the short duration of
interjection signals and significant inter- and intra-speaker variability. In
this work, we present and publish a dataset of interjection signals collected
specifically for interjection classification. We employ this dataset to train
and evaluate a baseline deep learning model. To enhance performance, we augment
the training dataset using techniques such as tempo and pitch transformation,
which significantly improve classification accuracy, making models more robust.
The interjection dataset, a Python library for the augmentation pipeline,
baseline model, and evaluation scripts, are available to the research
community.

### 8. [The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices](http://arxiv.org/pdf/2509.02910v1)

Authors: Sandra C. Matz, C. Blaine Horton, Sofie Goethals

Large language models (LLMs) increasingly act on people's behalf: they write
emails, buy groceries, and book restaurants. While the outsourcing of human
decision-making to AI can be both efficient and effective, it raises a
fundamental question: how does delegating identity-defining choices to AI
reshape who people become? We study the impact of agentic LLMs on two
identity-relevant outcomes: interpersonal distinctiveness - how unique a
person's choices are relative to others - and intrapersonal diversity - the
breadth of a single person's choices over time. Using real choices drawn from
social-media behavior of 1,000 U.S. users (110,000 choices in total), we
compare a generic and personalized agent to a human baseline. Both agents shift
people's choices toward more popular options, reducing the distinctiveness of
their behaviors and preferences. While the use of personalized agents tempers
this homogenization (compared to the generic AI), it also more strongly
compresses the diversity of people's preference portfolios by narrowing what
they explore across topics and psychological affinities. Understanding how AI
agents might flatten human experience, and how using generic versus
personalized agents involves distinctiveness-diversity trade-offs, is critical
for designing systems that augment rather than constrain human agency, and for
safeguarding diversity in thought, taste, and expression.

### 9. [Simulacra Naturae: Generative Ecosystem driven by Agent-Based Simulations and Brain Organoid Collective Intelligence](http://arxiv.org/pdf/2509.02924v1)

Authors: Nefeli Manoudaki, Mert Toka, Iason Paterakis, Diarmid Flatley

Simulacra Naturae is a data-driven media installation that explores
collective care through the entanglement of biological computation, material
ecologies, and generative systems. The work translates pre-recorded neural
activity from brain organoids, lab-grown three-dimensional clusters of neurons,
into a multi-sensory environment composed of generative visuals, spatial audio,
living plants, and fabricated clay artifacts. These biosignals, streamed
through a real-time system, modulate emergent agent behaviors inspired by
natural systems such as termite colonies and slime molds. Rather than using
biosignals as direct control inputs, Simulacra Naturae treats organoid activity
as a co-creative force, allowing neural rhythms to guide the growth, form, and
atmosphere of a generative ecosystem. The installation features computationally
fabricated clay prints embedded with solenoids, adding physical sound
resonances to the generative surround composition. The spatial environment,
filled with live tropical plants and a floor-level projection layer featuring
real-time generative AI visuals, invites participants into a sensory field
shaped by nonhuman cognition. By grounding abstract data in living materials
and embodied experience, Simulacra Naturae reimagines visualization as a
practice of care, one that decentralizes human agency and opens new spaces for
ethics, empathy, and ecological attunement within hybrid computational systems.

### 10. [The Role of Embodiment in Intuitive Whole-Body Teleoperation for Mobile Manipulation](http://arxiv.org/pdf/2509.03222v1)

Authors: Sophia Bianchi Moyen, Rickmer Krohn, Sophie Lueth, Kay Pompetzki, Jan Peters, Vignesh Prasad, Georgia Chalvatzaki

Intuitive Teleoperation interfaces are essential for mobile manipulation
robots to ensure high quality data collection while reducing operator workload.
A strong sense of embodiment combined with minimal physical and cognitive
demands not only enhances the user experience during large-scale data
collection, but also helps maintain data quality over extended periods. This
becomes especially crucial for challenging long-horizon mobile manipulation
tasks that require whole-body coordination. We compare two distinct robot
control paradigms: a coupled embodiment integrating arm manipulation and base
navigation functions, and a decoupled embodiment treating these systems as
separate control entities. Additionally, we evaluate two visual feedback
mechanisms: immersive virtual reality and conventional screen-based
visualization of the robot's field of view. These configurations were
systematically assessed across a complex, multi-stage task sequence requiring
integrated planning and execution. Our results show that the use of VR as a
feedback modality increases task completion time, cognitive workload, and
perceived effort of the teleoperator. Coupling manipulation and navigation
leads to a comparable workload on the user as decoupling the embodiments, while
preliminary experiments suggest that data acquired by coupled teleoperation
leads to better imitation learning performance. Our holistic view on intuitive
teleoperation interfaces provides valuable insight into collecting
high-quality, high-dimensional mobile manipulation data at scale with the human
operator in mind. Project
website:https://sophiamoyen.github.io/role-embodiment-wbc-moma-teleop/

### Information Retrieval

### 1. [Knowledge graph-based personalized multimodal recommendation fusion framework](http://arxiv.org/pdf/2509.02943v1)

Authors: Yu Fang

In the contemporary age characterized by information abundance, rapid
advancements in artificial intelligence have rendered recommendation systems
indispensable. Conventional recommendation methodologies based on collaborative
filtering or individual attributes encounter deficiencies in capturing nuanced
user interests. Knowledge graphs and multimodal data integration offer enhanced
representations of users and items with greater richness and precision. This
paper reviews existing multimodal knowledge graph recommendation frameworks,
identifying shortcomings in modal interaction and higher-order dependency
modeling. We propose the Cross-Graph Cross-Modal Mutual Information-Driven
Unified Knowledge Graph Learning and Recommendation Framework
(CrossGMMI-DUKGLR), which employs pre-trained visual-text alignment models for
feature extraction, achieves fine-grained modality fusion through multi-head
cross-attention, and propagates higher-order adjacency information via graph
attention networks.

### 2. [A Plug-and-play Model-agnostic Embedding Enhancement Approach for Explainable Recommendation](http://arxiv.org/pdf/2509.03130v1)

Authors: Yunqi Mi, Boyang Yan, Guoshuai Zhao, Jialie Shen, Xueming Qian

Existing multimedia recommender systems provide users with suggestions of
media by evaluating the similarities, such as games and movies. To enhance the
semantics and explainability of embeddings, it is a consensus to apply
additional information (e.g., interactions, contexts, popularity). However,
without systematic consideration of representativeness and value, the utility
and explainability of embedding drops drastically. Hence, we introduce RVRec, a
plug-and-play model-agnostic embedding enhancement approach that can improve
both personality and explainability of existing systems. Specifically, we
propose a probability-based embedding optimization method that uses a
contrastive loss based on negative 2-Wasserstein distance to learn to enhance
the representativeness of the embeddings. In addtion, we introduce a reweighing
method based on multivariate Shapley values strategy to evaluate and explore
the value of interactions and embeddings. Extensive experiments on multiple
backbone recommenders and real-world datasets show that RVRec can improve the
personalization and explainability of existing recommenders, outperforming
state-of-the-art baselines.

### 3. [OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search](http://arxiv.org/pdf/2509.03236v1)

Authors: Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, Huangyu Dai, Xing Xu, Tong Zhao, Mingcan Peng, XiaoYang Zheng, Cong Zhang, Qihang Zhao, Yuqing Ding, Chenyi Lei, Wenwu Ou, Han Li

Traditional e-commerce search systems employ multi-stage cascading
architectures (MCA) that progressively filter items through recall,
pre-ranking, and ranking stages. While effective at balancing computational
efficiency with business conversion, these systems suffer from fragmented
computation and optimization objective collisions across stages, which
ultimately limit their performance ceiling. To address these, we propose
\textbf{OneSearch}, the first industrial-deployed end-to-end generative
framework for e-commerce search. This framework introduces three key
innovations: (1) a Keyword-enhanced Hierarchical Quantization Encoding (KHQE)
module, to preserve both hierarchical semantics and distinctive item attributes
while maintaining strong query-item relevance constraints; (2) a multi-view
user behavior sequence injection strategy that constructs behavior-driven user
IDs and incorporates both explicit short-term and implicit long-term sequences
to model user preferences comprehensively; and (3) a Preference-Aware Reward
System (PARS) featuring multi-stage supervised fine-tuning and adaptive
reward-weighted ranking to capture fine-grained user preferences. Extensive
offline evaluations on large-scale industry datasets demonstrate OneSearch's
superior performance for high-quality recall and ranking. The rigorous online
A/B tests confirm its ability to enhance relevance in the same exposure
position, achieving statistically significant improvements: +1.67\% item CTR,
+2.40\% buyer, and +3.22\% order volume. Furthermore, OneSearch reduces
operational expenditure by 75.40\% and improves Model FLOPs Utilization from
3.26\% to 27.32\%. The system has been successfully deployed across multiple
search scenarios in Kuaishou, serving millions of users, generating tens of
millions of PVs daily.

### 4. [RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation](http://arxiv.org/pdf/2509.02942v1)

Authors: Renzhi Wu, Junjie Yang, Li Chen, Hong Li, Li Yu, Hong Yan

Cross-domain recommendation systems face the challenge of integrating
fine-grained user and item relationships across various product domains. To
address this, we introduce RankGraph, a scalable graph learning framework
designed to serve as a core component in recommendation foundation models
(FMs). By constructing and leveraging graphs composed of heterogeneous nodes
and edges across multiple products, RankGraph enables the integration of
complex relationships between users, posts, ads, and other entities. Our
framework employs a GPU-accelerated Graph Neural Network and contrastive
learning, allowing for dynamic extraction of subgraphs such as item-item and
user-user graphs to support similarity-based retrieval and real-time
clustering. Furthermore, RankGraph integrates graph-based pretrained
representations as contextual tokens into FM sequence models, enriching them
with structured relational knowledge. RankGraph has demonstrated improvements
in click (+0.92%) and conversion rates (+2.82%) in online A/B tests, showcasing
its effectiveness in cross-domain recommendation scenarios.

### 5. [Training LLMs to be Better Text Embedders through Bidirectional Reconstruction](http://arxiv.org/pdf/2509.03020v1)

Authors: Chang Su, Dengliang Shi, Siyuan Huang, Jintao Du, Changhua Meng, Yu Cheng, Weiqiang Wang, Zhouhan Lin

Large language models (LLMs) have increasingly been explored as powerful text
embedders. Existing LLM-based text embedding approaches often leverage the
embedding of the final token, typically a reserved special token such as [EOS].
However, these tokens have not been intentionally trained to capture the
semantics of the whole context, limiting their capacity as text embeddings,
especially for retrieval and re-ranking tasks. We propose to add a new training
stage before contrastive learning to enrich the semantics of the final token
embedding. This stage employs bidirectional generative reconstruction tasks,
namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based
Document-to-Query), which interleave to anchor the [EOS] embedding and
reconstruct either side of Query-Document pairs. Experimental results
demonstrate that our additional training stage significantly improves LLM
performance on the Massive Text Embedding Benchmark (MTEB), achieving new
state-of-the-art results across different LLM base models and scales.

### 6. [RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation](http://arxiv.org/pdf/2509.03131v1)

Authors: Sashuai Zhou, Weinan Gan, Qijiong Liu, Ke Lei, Jieming Zhu, Hai Huang, Yan Xia, Ruiming Tang, Zhenhua Dong, Zhou Zhao

Recent advances in LLM-based recommendation have shown promise, yet their
cross-domain generalization is hindered by a fundamental mismatch between
language-centric pretraining and the recommendation task. Existing methods,
relying on language-level knowledge, fail to capture dynamic, item-level user
interests across domains. To bridge this gap, we propose RecBase, a
domain-agnostic foundational model pretrained with a recommendation-oriented
objective. RecBase leverages a large-scale, heterogeneous, cross-domain corpus
with unified textual representations and feature mappings to enhance
cross-domain generalization. To further align item semantics across domains, we
introduce a unified item tokenizer that encodes items into hierarchical concept
identifiers, enabling structured representation and efficient vocabulary
sharing. The model is trained using an autoregressive objective to capture
complex item-level sequential patterns. On eight real-world datasets, our
1.5B-parameter model matches or surpasses the performance of LLM baselines up
to 7B parameters in zero-shot and cross-domain recommendation tasks.

### 7. [Enhancing Interpretability and Effectiveness in Recommendation with Numerical Features via Learning to Contrast the Counterfactual samples](http://arxiv.org/pdf/2509.03187v1)

Authors: Xiaoxiao Xu, Hao Wu, Wenhui Yu, Lantao Hu, Peng Jiang, Kun Gai

We propose a general model-agnostic Contrastive learning framework with
Counterfactual Samples Synthesizing (CCSS) for modeling the monotonicity
between the neural network output and numerical features which is critical for
interpretability and effectiveness of recommender systems. CCSS models the
monotonicity via a two-stage process: synthesizing counterfactual samples and
contrasting the counterfactual samples. The two techniques are naturally
integrated into a model-agnostic framework, forming an end-to-end training
process. Abundant empirical tests are conducted on a publicly available dataset
and a real industrial dataset, and the results well demonstrate the
effectiveness of our proposed CCSS. Besides, CCSS has been deployed in our real
large-scale industrial recommender, successfully serving over hundreds of
millions users.

### 8. [Knowledge Integration for Physics-informed Symbolic Regression Using Pre-trained Large Language Models](http://arxiv.org/pdf/2509.03036v1)

Authors: Bilge Taskin, Wenxiong Xie, Teddy Lazebnik

Symbolic regression (SR) has emerged as a powerful tool for automated
scientific discovery, enabling the derivation of governing equations from
experimental data. A growing body of work illustrates the promise of
integrating domain knowledge into the SR to improve the discovered equation's
generality and usefulness. Physics-informed SR (PiSR) addresses this by
incorporating domain knowledge, but current methods often require specialized
formulations and manual feature engineering, limiting their adaptability only
to domain experts. In this study, we leverage pre-trained Large Language Models
(LLMs) to facilitate knowledge integration in PiSR. By harnessing the
contextual understanding of LLMs trained on vast scientific literature, we aim
to automate the incorporation of domain knowledge, reducing the need for manual
intervention and making the process more accessible to a broader range of
scientific problems. Namely, the LLM is integrated into the SR's loss function,
adding a term of the LLM's evaluation of the SR's produced equation. We
extensively evaluate our method using three SR algorithms (DEAP, gplearn, and
PySR) and three pre-trained LLMs (Falcon, Mistral, and LLama 2) across three
physical dynamics (dropping ball, simple harmonic motion, and electromagnetic
wave). The results demonstrate that LLM integration consistently improves the
reconstruction of physical dynamics from data, enhancing the robustness of SR
models to noise and complexity. We further explore the impact of prompt
engineering, finding that more informative prompts significantly improve
performance.

### 9. [AI-Driven Drug Repurposing through miRNA-mRNA Relation](http://arxiv.org/pdf/2509.03336v1)

Authors: Sharanya Manoharan, Balu Bhasuran, Oviya Ramalakshmi Iyyappan, Mohamed Saleem Abdul Shukkoor, Malathi Sellapan, Kalpana Raja

miRNA mRNA relations are closely linked to several biological processes and
disease mechanisms In a recent study we tested the performance of large
language models LLMs on extracting miRNA mRNA relations from PubMed PubMedBERT
achieved the best performance of 0.783 F1 score for miRNA mRNA Interaction
Corpus MMIC Here we first applied the finetuned PubMedBERT model to extract
miRNA mRNA relations from PubMed for chronic obstructive pulmonary disease COPD
Alzheimers disease AD stroke type 2 diabetes mellitus T2DM chronic liver
disease and cancer Next we retrieved miRNA drug relations using KinderMiner a
literature mining tool for relation extraction Then we constructed three
interaction networks 1 disease centric network 2 drug centric network and 3
miRNA centric network comprising 3497 nodes and 16417 edges organized as a
directed graph to capture complex biological relationships Finally we validated
the drugs using MIMIC IV Our integrative approach revealed both established and
novel candidate drugs for diseases under study through 595 miRNA drug relations
extracted from PubMed To the best of our knowledge this is the first study to
systematically extract and visualize relationships among four distinct
biomedical entities miRNA mRNA drug and disease

### Machine Learning

### 1. [A Narrative Review of Clinical Decision Support Systems in Offloading Footwear for Diabetes-Related Foot Ulcers](http://arxiv.org/pdf/2509.02923v1)

Authors: Kunal Kumar, Muhammad Ashad Kabir, Luke Donnan, Sayed Ahmed

Offloading footwear helps prevent and treat diabetic foot ulcers (DFUs) by
lowering plantar pressure (PP), yet prescription decisions remain fragmented:
feature selection varies, personalization is limited, and evaluation practices
differ. We performed a narrative review of 45 studies (12 guidelines/protocols,
25 knowledge-based systems, 8 machine-learning applications) published to Aug
2025. We thematically analyzed knowledge type, decision logic, evaluation
methods, and enabling technologies. Guidelines emphasize PP thresholds (<=200
kPa or >=25--30\% reduction) but rarely yield actionable, feature-level
outputs. Knowledge-based systems use rule- and sensor-driven logic, integrating
PP monitoring, adherence tracking, and usability testing. ML work introduces
predictive, optimization, and generative models with high computational
accuracy but limited explainability and clinical validation. Evaluation remains
fragmented: protocols prioritize biomechanical tests; knowledge-based systems
assess usability/adherence; ML studies focus on technical accuracy with weak
linkage to long-term outcomes. From this synthesis we propose a five-part CDSS
framework: (1) a minimum viable dataset; (2) a hybrid architecture combining
rules, optimization, and explainable ML; (3) structured feature-level outputs;
(4) continuous validation and evaluation; and (5) integration with clinical and
telehealth workflows. This framework aims to enable scalable, patient-centered
CDSSs for DFU care; prioritizing interoperable datasets, explainable models,
and outcome-focused evaluation will be key to clinical adoption.

### 2. [Multimodal learning of melt pool dynamics in laser powder bed fusion](http://arxiv.org/pdf/2509.03029v1)

Authors: Satyajit Mojumder, Pallock Halder, Tiana Tonge

While multiple sensors are used for real-time monitoring in additive
manufacturing, not all provide practical or reliable process insights. For
example, high-speed X-ray imaging offers valuable spatial information about
subsurface melt pool behavior but is costly and impractical for most industrial
settings. In contrast, absorptivity data from low-cost photodiodes correlate
with melt pool dynamics but is often too noisy for accurate prediction when
used alone. In this paper, we propose a multimodal data fusion approach for
predicting melt pool dynamics by combining high-fidelity X-ray data with
low-fidelity absorptivity data in the Laser Powder Bed Fusion (LPBF) process.
Our multimodal learning framework integrates convolutional neural networks
(CNNs) for spatial feature extraction from X-ray data with recurrent neural
networks (RNNs) for temporal feature extraction from absorptivity signals,
using an early fusion strategy. The multimodal model is further used as a
transfer learning model to fine-tune the RNN model that can predict melt pool
dynamics only with absorptivity, with greater accuracy compared to the
multimodal model. Results show that training with both modalities significantly
improves prediction accuracy compared to using either modality alone.
Furthermore, once trained, the model can infer melt pool characteristics using
only absorptivity data, eliminating the need for expensive X-ray imaging. This
multimodal fusion approach enables cost-effective, real-time monitoring and has
broad applicability in additive manufacturing.

### 3. [Discrete Functional Geometry of ReLU Networks via ReLU Transition Graphs](http://arxiv.org/pdf/2509.03056v1)

Authors: Sahil Rajesh Dhayalkar

We extend the ReLU Transition Graph (RTG) framework into a comprehensive
graph-theoretic model for understanding deep ReLU networks. In this model, each
node represents a linear activation region, and edges connect regions that
differ by a single ReLU activation flip, forming a discrete geometric structure
over the network's functional behavior. We prove that RTGs at random
initialization exhibit strong expansion, binomial degree distributions, and
spectral properties that tightly govern generalization. These structural
insights enable new bounds on capacity via region entropy and on generalization
via spectral gap and edge-wise KL divergence. Empirically, we construct RTGs
for small networks, measure their smoothness and connectivity properties, and
validate theoretical predictions. Our results show that region entropy
saturates under overparameterization, spectral gap correlates with
generalization, and KL divergence across adjacent regions reflects functional
smoothness. This work provides a unified framework for analyzing ReLU networks
through the lens of discrete functional geometry, offering new tools to
understand, diagnose, and improve generalization.

### 4. [Systematic Evaluation of Attribution Methods: Eliminating Threshold Bias and Revealing Method-Dependent Performance Patterns](http://arxiv.org/pdf/2509.03176v1)

Authors: Serra Aksoy

Attribution methods explain neural network predictions by identifying
influential input features, but their evaluation suffers from threshold
selection bias that can reverse method rankings and undermine conclusions.
Current protocols binarize attribution maps at single thresholds, where
threshold choice alone can alter rankings by over 200 percentage points. We
address this flaw with a threshold-free framework that computes Area Under the
Curve for Intersection over Union (AUC-IoU), capturing attribution quality
across the full threshold spectrum. Evaluating seven attribution methods on
dermatological imaging, we show single-threshold metrics yield contradictory
results, while threshold-free evaluation provides reliable differentiation.
XRAI achieves 31% improvement over LIME and 204% over vanilla Integrated
Gradients, with size-stratified analysis revealing performance variations up to
269% across lesion scales. These findings establish methodological standards
that eliminate evaluation artifacts and enable evidence-based method selection.
The threshold-free framework provides both theoretical insight into attribution
behavior and practical guidance for robust comparison in medical imaging and
beyond.

### 5. [Tabular foundation model for GEOAI benchmark problems BM/AirportSoilProperties/2/2025](http://arxiv.org/pdf/2509.03191v1)

Authors: Taiga Saito, Yu Otake, Stephen Wu

This paper presents a novel application of the Tabular Prior-Data Fitted
Network (TabPFN) - a transformer-based foundation model for tabular data - to
geotechnical site characterization problems defined in the GEOAI benchmark
BM/AirportSoilProperties/2/2025. Two tasks are addressed: (1) predicting the
spatial variation of undrained shear strength (su) across borehole depth
profiles, and (2) imputing missing mechanical parameters in a dense-site
dataset. We apply TabPFN in a zero-training, few-shot, in-context learning
setting - without hyper-parameter tuning - and provide it with additional
context from the big indirect database (BID). The study demonstrates that
TabPFN, as a general-purpose foundation model, achieved superior accuracy and
well-calibrated predictive distributions compared to a conventional
hierarchical Bayesian model (HBM) baseline, while also offering significant
gains in inference efficiency. In Benchmark Problem #1 (spatial su prediction),
TabPFN outperformed the HBM in prediction accuracy and delivered an
order-of-magnitude faster runtime. In Benchmark Problem #2 (missing mechanical
parameter imputation), TabPFN likewise achieved lower RMSE for all target
parameters with well-quantified uncertainties, though its cumulative
computation cost was higher than HBM's due to its one-variable-at-a-time
inference. These results mark the first successful use of a tabular foundation
model in geotechnical modeling, suggesting a potential paradigm shift in
probabilistic site characterization.

### 6. [Exploring the Design Space of Fair Tree Learning Algorithms](http://arxiv.org/pdf/2509.03204v1)

Authors: Kiara Stempel, Mattia Cerrato, Stefan Kramer

Decision trees have been studied extensively in the context of fairness,
aiming to maximize prediction performance while ensuring non-discrimination
against different groups. Techniques in this space usually focus on imposing
constraints at training time, constraining the search space so that solutions
which display unacceptable values of relevant metrics are not considered,
discarded, or discouraged. If we assume one target variable y and one sensitive
attribute s, the design space of tree learning algorithms can be spanned as
follows: (i) One can have one tree T that is built using an objective function
that is a function of y, s, and T. For instance, one can build a tree based on
the weighted information gain regarding y (maximizing) and s (minimizing). (ii)
The second option is to have one tree model T that uses an objective function
in y and T and a constraint on s and T. Here, s is no longer part of the
objective, but part of a constraint. This can be achieved greedily by aborting
a further split as soon as the condition that optimizes the objective in y
fails to satisfy the constraint on s. A simple way to explore other splits is
to backtrack during tree construction once a fairness constraint is violated.
(iii) The third option is to have two trees T_y and T_s, one for y and one for
s, such that the tree structure for y and s does not have to be shared. In this
way, information regarding y and regarding s can be used independently, without
having to constrain the choices in tree construction by the mutual information
between the two variables. Quite surprisingly, of the three options, only the
first one and the greedy variant of the second have been studied in the
literature so far. In this paper, we introduce the above two additional options
from that design space and characterize them experimentally on multiple
datasets.

### 7. [TeRA: Vector-based Random Tensor Network for High-Rank Adaptation of Large Language Models](http://arxiv.org/pdf/2509.03234v1)

Authors: Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, Danilo Mandic

Parameter-Efficient Fine-Tuning (PEFT) methods, such as Low-Rank Adaptation
(LoRA), have significantly reduced the number of trainable parameters needed in
fine-tuning large language models (LLMs). Subsequent developments of LoRA-style
adapters have diverged into two main directions: (1) enhancing model
expressivity with high-rank adapters, and (2) pushing for further parameter
reduction, as exemplified by vector-based methods. However, these approaches
present a trade-off, as achieving the expressivity of high-rank weight updates
typically comes at the cost of sacrificing the extreme parameter efficiency
offered by vector-based techniques. To address this issue, we propose a
vector-based random \underline{\textbf{Te}}nsor network for
high-\underline{\textbf{R}}ank \underline{\textbf{A}}daptation (TeRA), a novel
PEFT method that achieves high-rank weight updates while retaining the
parameter efficiency of vector-based PEFT adapters. This is achieved by
parameterizing the tensorized weight update matrix as a Tucker-like tensor
network (TN), in which large randomly initialized factors are frozen and shared
across layers, while only small layer-specific scaling vectors, formed by
entries in diagonal factor matrices, are trained. This design effectively
decouples the rank of the weight update matrix from the number of trainable
parameters. Comprehensive experiments demonstrate that TeRA matches or even
outperforms high-rank adapters, while requiring a trainable parameter count
similar to vector-based methods. Theoretical analysis and ablation studies
further validate the effectiveness of our approach.

### 8. [Unsupervised Learning based Element Resource Allocation for Reconfigurable Intelligent Surfaces in mmWave Network](http://arxiv.org/pdf/2509.03241v1)

Authors: Pujitha Mamillapalli, Yoghitha Ramamoorthi, Abhinav Kumar, Tomoki Murakami, Tomoaki Ogawa, Yasushi Takatori

The increasing demand for high data rates and seamless connectivity in
wireless systems has sparked significant interest in reconfigurable intelligent
surfaces (RIS) and artificial intelligence-based wireless applications. RIS
typically comprises passive reflective antenna elements that control the
wireless propagation environment by adequately tuning the phase of the
reflective elements. The allocation of RIS elements to multipleuser equipment
(UEs) is crucial for efficiently utilizing RIS. In this work, we formulate a
joint optimization problem that optimizes the RIS phase configuration and
resource allocation under an $\alpha$-fair scheduling framework and propose an
efficient way of allocating RIS elements. Conventional iterative optimization
methods, however, suffer from exponentially increasing computational complexity
as the number of RIS elements increases and also complicate the generation of
training labels for supervised learning. To overcome these challenges, we
propose a five-layer fully connected neural network (FNN) combined with a
preprocessing technique to significantly reduce input dimensionality, lower
computational complexity, and enhance scalability. The simulation results show
that our proposed NN-based solution reduces computational overhead while
significantly improving system throughput by 6.8% compared to existing RIS
element allocation schemes. Furthermore, the proposed system achieves better
performance while reducing computational complexity, making it significantly
more scalable than the iterative optimization algorithms.

### 9. [Meta-Imputation Balanced (MIB): An Ensemble Approach for Handling Missing Data in Biomedical Machine Learning](http://arxiv.org/pdf/2509.03316v1)

Authors: Fatemeh Azad, Zoran Bosnić, Matjaž Kukar

Missing data represents a fundamental challenge in machine learning
applications, often reducing model performance and reliability. This problem is
particularly acute in fields like bioinformatics and clinical machine learning,
where datasets are frequently incomplete due to the nature of both data
generation and data collection. While numerous imputation methods exist, from
simple statistical techniques to advanced deep learning models, no single
method consistently performs well across diverse datasets and missingness
mechanisms. This paper proposes a novel Meta-Imputation approach that learns to
combine the outputs of multiple base imputers to predict missing values more
accurately. By training the proposed method called Meta-Imputation Balanced
(MIB) on synthetically masked data with known ground truth, the system learns
to predict the most suitable imputed value based on the behavior of each
method. Our work highlights the potential of ensemble learning in imputation
and paves the way for more robust, modular, and interpretable preprocessing
pipelines in real-world machine learning systems.

### 10. [EvolveSignal: A Large Language Model Powered Coding Agent for Discovering Traffic Signal Control Algorithms](http://arxiv.org/pdf/2509.03335v1)

Authors: Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, Zhenliang Ma

In traffic engineering, the fixed-time traffic signal control remains widely
used for its low cost, stability, and interpretability. However, its design
depends on hand-crafted formulas (e.g., Webster) and manual re-timing by
engineers to adapt to demand changes, which is labor-intensive and often yields
suboptimal results under heterogeneous or congested conditions. This paper
introduces the EvolveSignal, a large language models (LLMs) powered coding
agent to automatically discover new traffic signal control algorithms. We
formulate the problem as program synthesis, where candidate algorithms are
represented as Python functions with fixed input-output structures, and
iteratively optimized through external evaluations (e.g., a traffic simulator)
and evolutionary search. Experiments on a signalized intersection demonstrate
that the discovered algorithms outperform Webster's baseline, reducing average
delay by 20.1% and average stops by 47.1%. Beyond performance, ablation and
incremental analyses reveal that EvolveSignal modifications-such as adjusting
cycle length bounds, incorporating right-turn demand, and rescaling green
allocations-can offer practically meaningful insights for traffic engineers.
This work opens a new research direction by leveraging AI for algorithm design
in traffic signal control, bridging program synthesis with transportation
engineering.

### Neural and Evolutionary Computing

### 1. [A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks](http://arxiv.org/pdf/2509.03281v1)

Authors: Qianyi Bai, Haiteng Wang, Qiang Yu

While spiking neural networks (SNNs) provide a biologically inspired and
energy-efficient computational framework, their robustness and the dynamic
advantages inherent to biological neurons remain significantly underutilized
owing to oversimplified neuron models. In particular, conventional leaky
integrate-and-fire (LIF) neurons often omit the dynamic conductance mechanisms
inherent in biological neurons, thereby limiting their capacity to cope with
noise and temporal variability. In this work, we revisit dynamic conductance
from a functional perspective and uncover its intrinsic role as a biologically
plausible gating mechanism that modulates information flow. Building on this
insight, we introduce the Dynamic Gated Neuron~(DGN), a novel spiking unit in
which membrane conductance evolves in response to neuronal activity, enabling
selective input filtering and adaptive noise suppression. We provide a
theoretical analysis showing that DGN possess enhanced stochastic stability
compared to standard LIF models, with dynamic conductance intriguingly acting
as a disturbance rejection mechanism. DGN-based SNNs demonstrate superior
performance across extensive evaluations on anti-noise tasks and
temporal-related benchmarks such as TIDIGITS and SHD, consistently exhibiting
excellent robustness. Our results highlight, for the first time, a biologically
plausible dynamic gating as a key mechanism for robust spike-based computation,
providing not only theoretical guarantees but also strong empirical
validations. This work thus paves the way for more resilient, efficient, and
biologically inspired spiking neural networks.

### 2. [StableSleep: Source-Free Test-Time Adaptation for Sleep Staging with Lightweight Safety Rails](http://arxiv.org/pdf/2509.02982v1)

Authors: Hritik Arasu, Faisal R Jahangiri

Sleep staging models often degrade when deployed on patients with unseen
physiology or recording conditions. We propose a streaming, source-free
test-time adaptation (TTA) recipe that combines entropy minimization (Tent)
with Batch-Norm statistic refresh and two safety rails: an entropy gate to
pause adaptation on uncertain windows and an EMA-based reset to reel back
drift. On Sleep-EDF Expanded, using single-lead EEG (Fpz-Cz, 100 Hz, 30s
epochs; R&K to AASM mapping), we show consistent gains over a frozen baseline
at seconds-level latency and minimal memory, reporting per-stage metrics and
Cohen's k. The method is model-agnostic, requires no source data or patient
calibration, and is practical for on-device or bedside use.

### 3. [Decentralised self-organisation of pivoting cube ensembles using geometric deep learning](http://arxiv.org/pdf/2509.03140v1)

Authors: Nadezhda Dobreva, Emmanuel Blazquez, Jai Grover, Dario Izzo, Yuzhen Qin, Dominik Dold

We present a decentralized model for autonomous reconfiguration of
homogeneous pivoting cube modular robots in two dimensions. Each cube in the
ensemble is controlled by a neural network that only gains information from
other cubes in its local neighborhood, trained using reinforcement learning.
Furthermore, using geometric deep learning, we include the grid symmetries of
the cube ensemble in the neural network architecture. We find that even the
most localized versions succeed in reconfiguring to the target shape, although
reconfiguration happens faster the more information about the whole ensemble is
available to individual cubes. Near-optimal reconfiguration is achieved with
only nearest neighbor interactions by using multiple information passing
between cubes, allowing them to accumulate more global information about the
ensemble. Compared to standard neural network architectures, using geometric
deep learning approaches provided only minor benefits. Overall, we successfully
demonstrate mostly local control of a modular self-assembling system, which is
transferable to other space-relevant systems with different action spaces, such
as sliding cube modular robots and CubeSat swarms.

### Networking and Internet Architecture

### 1. [Hierarchical Low-Altitude Wireless Network Empowered Air Traffic Management](http://arxiv.org/pdf/2509.03386v1)

Authors: Ziye Jia, Jia He, Yuanhao Cui, Qiuming Zhu, Ligang Yuan, Fuhui Zhou, Qihui Wu, Dusit Niyato, Zhu Han

As the increasing development of low-altitude aircrafts, the rational design
of low-altitude networks directly impacts the aerial safety and resource
utilization. To address the challenges of environmental complexity and aircraft
diversity in the traffic management, we propose a hierarchical low-altitude
wireless network (HLWN) framework. Empowered by the threedimensional spatial
discretization and integrated wireless monitoring mechanisms in HLWN, we design
low-altitude air corridors to guarantee safe operation and optimization.
Besides, we develop the multi-dimensional flight risk assessment through
conflict detection and probabilistic collision analysis, facilitating dynamic
collision avoidance for heterogeneous aircrafts. Finally, the open issues and
future directions are investigated to provide insights into HLAN development.

### 2. [Closing the Visibility Gap: A Monitoring Framework for Verifiable Open RAN Operations](http://arxiv.org/pdf/2509.03000v1)

Authors: Hexuan Yu, Md Mohaimin Al Barat, Yang Xiao, Y. Thomas Hou, Wenjing Lou

Open Radio Access Network (Open RAN) is reshaping mobile network architecture
by promoting openness, disaggregation, and cross-vendor interoperability.
However, this architectural flexibility introduces new security challenges,
especially in deployments where multiple mobile network operators (MNOs)
jointly operate shared components. Existing Zero Trust Architectures (ZTA) in
O-RAN, as defined by governmental and industry standards, implicitly assume
that authenticated components will comply with operational policies. However,
this assumption creates a critical blind spot: misconfigured or compromised
components can silently violate policies, misuse resources, or corrupt
downstream processes (e.g., ML-based RIC xApps).
  To address this critical gap, we propose a monitoring framework for low-trust
O-RAN environments that proactively verifies configuration state and control
behavior against tenant-defined policies. Our system provides scalable,
verifiable oversight to enhance transparency and trust in O-RAN operations. We
implement and evaluate the framework using standardized O-RAN configurations,
with total processing latency of approximately 200 ms, demonstrating its
efficiency and practicality for timely policy enforcement and compliance
auditing in multi-MNO deployments.

### 3. [Dependency Chain Analysis of ROS 2 DDS QoS Policies: From Lifecycle Tutorial to Static Verification](http://arxiv.org/pdf/2509.03381v1)

Authors: Sanghoon Lee, Junha Kang, Kyung-Joon Park

Robot Operating System 2 (ROS 2) relies on the Data Distribution Service
(DDS), which offers more than 20 Quality of Service (QoS) policies governing
availability, reliability, and resource usage. Yet ROS 2 users lack clear
guidance on safe policy combinations and validation processes prior to
deployment, which often leads to trial-and-error tuning and unexpected runtime
failures. To address these challenges, we analyze DDS Publisher-Subscriber
communication over a life cycle divided into Discovery, Data Exchange, and
Disassociation, and provide a user oriented tutorial explaining how 16 QoS
policies operate in each phase. Building on this analysis, we derive a QoS
dependency chain that formalizes inter-policy relationships and classifies 41
dependency violation rules, capturing constraints that commonly cause
communication failures in practice. Finally, we introduce QoS Guard, a ROS 2
package that statically validates DDS XML profiles offline, flags conflicts,
and enables safe, predeployment tuning without establishing a live ROS 2
session. Together, these contributions give ROS 2 users both conceptual insight
and a concrete tool that enables early detection of misconfigurations,
improving the reliability and resource efficiency of ROS 2 based robotic
systems.

### 4. [Multi-layer Digital Twin System for Future Mobile Metaverse](http://arxiv.org/pdf/2509.03049v1)

Authors: Gaosheng Zhao, Dong In Kim

In the upcoming 6G era, the communication networks are expected to face
unprecedented challenges in terms of complexity and dynamics. Digital Twin (DT)
technology, with its various digital capabilities, holds great potential to
facilitate the transformation of the communication network from passive
responding to proactive adaptation. Thus, in this paper, we propose a
multi-layer DT system that coordinates local DT, edge DT, and cloud DT for
future network architecture and functions. In our vision, the proposed DT
system will not only achieve real-time data-driven decision-making and digital
agent functions previously handled by centralized DT, but will do so in a more
distributed, mobile, layer-by-layer manner. Moreover, it will supply essential
data, pre-trained models, and open interfaces for future metaverse
applications, enabling creators and users to efficiently develop and experience
metaverse services.

### 5. [Machine Learning-Driven Anomaly Detection for 5G O-RAN Performance Metrics](http://arxiv.org/pdf/2509.03290v1)

Authors: Babak Azkaei, Kishor Chandra Joshi, George Exarchakos

The ever-increasing reliance of critical services on network infrastructure
coupled with the increased operational complexity of beyond-5G/6G networks
necessitate the need for proactive and automated network fault management. The
provision for open interfaces among different radio access network\,(RAN)
elements and the integration of AI/ML into network architecture enabled by the
Open RAN\,(O-RAN) specifications bring new possibilities for active network
health monitoring and anomaly detection. In this paper we leverage these
advantages and develop an anomaly detection framework that proactively detect
the possible throughput drops for a UE and minimize the post-handover failures.
We propose two actionable anomaly detection algorithms tailored for real-world
deployment. The first algorithm identifies user equipment (UE) at risk of
severe throughput degradation by analyzing key performance indicators (KPIs)
such as resource block utilization and signal quality metrics, enabling
proactive handover initiation. The second algorithm evaluates neighbor cell
radio coverage quality, filtering out cells with anomalous signal strength or
interference levels. This reduces candidate targets for handover by 41.27\% on
average. Together, these methods mitigate post-handover failures and throughput
drops while operating much faster than the near-real-time latency constraints.
This paves the way for self-healing 6G networks.

### Robotics

### 1. [IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments](http://arxiv.org/pdf/2509.02972v1)

Authors: Haolan Zhang, Thanh Nguyen Canh, Chenghao Li, Ruidong Yang, Yonghoon Ji, Nak Young Chong

Visual Simultaneous Localization and Mapping (SLAM) plays a crucial role in
autonomous systems. Traditional SLAM methods, based on static environment
assumptions, struggle to handle complex dynamic environments. Recent dynamic
SLAM systems employ geometric constraints and deep learning to remove dynamic
features, yet this creates a new challenge: insufficient remaining point
features for subsequent SLAM processes. Existing solutions address this by
continuously introducing additional line and plane features to supplement point
features, achieving robust tracking and pose estimation. However, current
methods continuously introduce additional features regardless of necessity,
causing two problems: unnecessary computational overhead and potential
performance degradation from accumulated low-quality additional features and
noise. To address these issues, this paper proposes a feature-aware mechanism
that evaluates whether current features are adequate to determine if line
feature support should be activated. This decision mechanism enables the system
to introduce line features only when necessary, significantly reducing
computational complexity of additional features while minimizing the
introduction of low-quality features and noise. In subsequent processing, the
introduced line features assist in obtaining better initial camera poses
through tracking, local mapping, and loop closure, but are excluded from global
optimization to avoid potential negative impacts from low-quality additional
features in long-term process. Extensive experiments on TUM datasets
demonstrate substantial improvements in both ATE and RPE metrics compared to
ORB-SLAM3 baseline and superior performance over other dynamic SLAM and
multi-feature methods.

### 2. [CTBC: Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning](http://arxiv.org/pdf/2509.02986v1)

Authors: Rankun Li, Hao Wang, Qi Li, Zhuo Han, Yifei Chu, Linqi Ye, Wende Xie, Wenlong Liao

In recent years, wheeled bipedal robots have gained increasing attention due
to their advantages in mobility, such as high-speed locomotion on flat terrain.
However, their performance on complex environments (e.g., staircases) remains
inferior to that of traditional legged robots. To overcome this limitation, we
propose a general contact-triggered blind climbing (CTBC) framework for wheeled
bipedal robots. Upon detecting wheel-obstacle contact, the robot triggers a
leg-lifting motion to overcome the obstacle. By leveraging a strongly-guided
feedforward trajectory, our method enables the robot to rapidly acquire agile
leg-lifting skills, significantly enhancing its capability to traverse
unstructured terrains. The approach has been experimentally validated and
successfully deployed on LimX Dynamics' wheeled bipedal robot, Tron1.
Real-world tests demonstrate that Tron1 can reliably climb obstacles well
beyond its wheel radius using only proprioceptive feedback.

### 3. [Exploring persuasive Interactions with generative social robots: An experimental framework](http://arxiv.org/pdf/2509.03231v1)

Authors: Stephan Vonschallen, Larissa Julia Corina Finsler, Theresa Schmiedel, Friederike Eyssel

Integrating generative AI such as large language models into social robots
has improved their ability to engage in natural, human-like communication. This
study presents a method to examine their persuasive capabilities. We designed
an experimental framework focused on decision making and tested it in a pilot
that varied robot appearance and self-knowledge. Using qualitative analysis, we
evaluated interaction quality, persuasion effectiveness, and the robot's
communicative strategies. Participants generally experienced the interaction
positively, describing the robot as competent, friendly, and supportive, while
noting practical limits such as delayed responses and occasional
speech-recognition errors. Persuasiveness was highly context dependent and
shaped by robot behavior: participants responded well to polite, reasoned
suggestions and expressive gestures, but emphasized the need for more
personalized, context-aware arguments and clearer social roles. These findings
suggest that generative social robots can influence user decisions, but their
effectiveness depends on communicative nuance and contextual relevance. We
propose refinements to the framework to further study persuasive dynamics
between robots and human users.

### 4. [DUViN: Diffusion-Based Underwater Visual Navigation via Knowledge-Transferred Depth Features](http://arxiv.org/pdf/2509.02983v1)

Authors: Jinghe Yang, Minh-Quan Le, Mingming Gong, Ye Pu

Autonomous underwater navigation remains a challenging problem due to limited
sensing capabilities and the difficulty of constructing accurate maps in
underwater environments. In this paper, we propose a Diffusion-based Underwater
Visual Navigation policy via knowledge-transferred depth features, named DUViN,
which enables vision-based end-to-end 4-DoF motion control for underwater
vehicles in unknown environments. DUViN guides the vehicle to avoid obstacles
and maintain a safe and perception awareness altitude relative to the terrain
without relying on pre-built maps. To address the difficulty of collecting
large-scale underwater navigation datasets, we propose a method that ensures
robust generalization under domain shifts from in-air to underwater
environments by leveraging depth features and introducing a novel model
transfer strategy. Specifically, our training framework consists of two phases:
we first train the diffusion-based visual navigation policy on in-air datasets
using a pre-trained depth feature extractor. Secondly, we retrain the extractor
on an underwater depth estimation task and integrate the adapted extractor into
the trained navigation policy from the first step. Experiments in both
simulated and real-world underwater environments demonstrate the effectiveness
and generalization of our approach. The experimental videos are available at
https://www.youtube.com/playlist?list=PLqt2s-RyCf1gfXJgFzKjmwIqYhrP4I-7Y.

### 5. [Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression](http://arxiv.org/pdf/2509.03012v1)

Authors: Uddeshya Upadhyay

Deep neural networks (DNNs) are increasingly being used in autonomous
systems. However, DNNs do not generalize well to domain shift. Adapting to a
continuously evolving environment is a safety-critical challenge inevitably
faced by all autonomous systems deployed to the real world. Recent work on
test-time training proposes methods that adapt to a new test distribution on
the fly by optimizing the DNN model for each test input using self-supervision.
However, these techniques result in a sharp increase in inference time as
multiple forward and backward passes are required for a single test sample (for
test-time training) before finally making the prediction based on the
fine-tuned features. This is undesirable for real-world robotics applications
where these models may be deployed to resource constraint hardware with strong
latency requirements. In this work, we propose a new framework (called UT$^3$)
that leverages test-time training for improved performance in the presence of
continuous domain shift while also decreasing the inference time, making it
suitable for real-world applications. Our method proposes an uncertainty-aware
self-supervision task for efficient test-time training that leverages the
quantified uncertainty to selectively apply the training leading to sharp
improvements in the inference time while performing comparably to standard
test-time training protocol. Our proposed protocol offers a continuous setting
to identify the selected keyframes, allowing the end-user to control how often
to apply test-time training. We demonstrate the efficacy of our method on a
dense regression task - monocular depth estimation.

### 6. [Efficient Active Training for Deep LiDAR Odometry](http://arxiv.org/pdf/2509.03211v1)

Authors: Beibei Zhou, Zhiyuan Zhang, Zhenbo Song, Jianhui Guo, Hui Kong

Robust and efficient deep LiDAR odometry models are crucial for accurate
localization and 3D reconstruction, but typically require extensive and diverse
training data to adapt to diverse environments, leading to inefficiencies. To
tackle this, we introduce an active training framework designed to selectively
extract training data from diverse environments, thereby reducing the training
load and enhancing model generalization. Our framework is based on two key
strategies: Initial Training Set Selection (ITSS) and Active Incremental
Selection (AIS). ITSS begins by breaking down motion sequences from general
weather into nodes and edges for detailed trajectory analysis, prioritizing
diverse sequences to form a rich initial training dataset for training the base
model. For complex sequences that are difficult to analyze, especially under
challenging snowy weather conditions, AIS uses scene reconstruction and
prediction inconsistency to iteratively select training samples, refining the
model to handle a wide range of real-world scenarios. Experiments across
datasets and weather conditions validate our approach's effectiveness. Notably,
our method matches the performance of full-dataset training with just 52\% of
the sequence volume, demonstrating the training efficiency and robustness of
our active training paradigm. By optimizing the training process, our approach
sets the stage for more agile and reliable LiDAR odometry systems, capable of
navigating diverse environmental conditions with greater precision.

### 7. [AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation](http://arxiv.org/pdf/2509.03270v1)

Authors: Martin Skoglund, Fredrik Warg, Aria Mirzai, Anders Thorsen, Karl Lundgren, Peter Folkesson, Bastian Havers-zulka

Integrating Artificial Intelligence (AI) technology in electric vehicles (EV)
introduces unique challenges for safety assurance, particularly within the
framework of ISO 26262, which governs functional safety in the automotive
domain. Traditional assessment methodologies are not geared toward evaluating
AI-based functions and require evolving standards and practices. This paper
explores how an independent assessment of an AI component in an EV can be
achieved when combining ISO 26262 with the recently released ISO/PAS 8800,
whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC)
battery estimation exemplifies the process. Key features relevant to the
independent assessment of this extended evaluation approach are identified. As
part of the evaluation, robustness testing of the AI component is conducted
using fault injection experiments, wherein perturbed sensor inputs are
systematically introduced to assess the component's resilience to input
variance.

### 8. [Dependency Chain Analysis of ROS 2 DDS QoS Policies: From Lifecycle Tutorial to Static Verification](http://arxiv.org/pdf/2509.03381v1)

Authors: Sanghoon Lee, Junha Kang, Kyung-Joon Park

Robot Operating System 2 (ROS 2) relies on the Data Distribution Service
(DDS), which offers more than 20 Quality of Service (QoS) policies governing
availability, reliability, and resource usage. Yet ROS 2 users lack clear
guidance on safe policy combinations and validation processes prior to
deployment, which often leads to trial-and-error tuning and unexpected runtime
failures. To address these challenges, we analyze DDS Publisher-Subscriber
communication over a life cycle divided into Discovery, Data Exchange, and
Disassociation, and provide a user oriented tutorial explaining how 16 QoS
policies operate in each phase. Building on this analysis, we derive a QoS
dependency chain that formalizes inter-policy relationships and classifies 41
dependency violation rules, capturing constraints that commonly cause
communication failures in practice. Finally, we introduce QoS Guard, a ROS 2
package that statically validates DDS XML profiles offline, flags conflicts,
and enables safe, predeployment tuning without establishing a live ROS 2
session. Together, these contributions give ROS 2 users both conceptual insight
and a concrete tool that enables early detection of misconfigurations,
improving the reliability and resource efficiency of ROS 2 based robotic
systems.

### 9. [ANNIE: Be Careful of Your Robots](http://arxiv.org/pdf/2509.03383v1)

Authors: Yiyang Huang, Zixuan Wang, Zishen Wan, Yapeng Tian, Haobo Xu, Yinhe Han, Yiming Gan

The integration of vision-language-action (VLA) models into embodied AI (EAI)
robots is rapidly advancing their ability to perform complex, long-horizon
tasks in humancentric environments. However, EAI systems introduce critical
security risks: a compromised VLA model can directly translate adversarial
perturbations on sensory input into unsafe physical actions. Traditional safety
definitions and methodologies from the machine learning community are no longer
sufficient. EAI systems raise new questions, such as what constitutes safety,
how to measure it, and how to design effective attack and defense mechanisms in
physically grounded, interactive settings. In this work, we present the first
systematic study of adversarial safety attacks on embodied AI systems, grounded
in ISO standards for human-robot interactions. We (1) formalize a principled
taxonomy of safety violations (critical, dangerous, risky) based on physical
constraints such as separation distance, velocity, and collision boundaries;
(2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with
2,400 video-action sequences for evaluating embodied safety; and (3)
ANNIE-Attack, a task-aware adversarial framework with an attack leader model
that decomposes long-horizon goals into frame-level perturbations. Our
evaluation across representative EAI models shows attack success rates
exceeding 50% across all safety categories. We further demonstrate sparse and
adaptive attack strategies and validate the real-world impact through physical
robot experiments. These results expose a previously underexplored but highly
consequential attack surface in embodied AI systems, highlighting the urgent
need for security-driven defenses in the physical AI era. Code is available at
https://github.com/RLCLab/Annie.

### 10. [Real-Time Instrument Planning and Perception for Novel Measurements of Dynamic Phenomena](http://arxiv.org/pdf/2509.03500v1)

Authors: Itai Zilberstein, Alberto Candela, Steve Chien

Advancements in onboard computing mean remote sensing agents can employ
state-of-the-art computer vision and machine learning at the edge. These
capabilities can be leveraged to unlock new rare, transient, and pinpoint
measurements of dynamic science phenomena. In this paper, we present an
automated workflow that synthesizes the detection of these dynamic events in
look-ahead satellite imagery with autonomous trajectory planning for a
follow-up high-resolution sensor to obtain pinpoint measurements. We apply this
workflow to the use case of observing volcanic plumes. We analyze
classification approaches including traditional machine learning algorithms and
convolutional neural networks. We present several trajectory planning
algorithms that track the morphological features of a plume and integrate these
algorithms with the classifiers. We show through simulation an order of
magnitude increase in the utility return of the high-resolution instrument
compared to baselines while maintaining efficient runtimes.

### Software Engineering

### 1. [The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams](http://arxiv.org/pdf/2509.03463v1)

Authors: Parham Khamsepour, Mark Cole, Ish Ashraf, Sandeep Puri, Mehrdad Sabetzadeh, Shiva Nejati

Large Language Models (LLMs) show strong potential for automating the
generation of models from natural-language descriptions. A common approach is
an iterative generate-critique-refine loop, where candidate models are
produced, evaluated, and updated based on detected issues. This process needs
to address: (1) structural correctness - compliance with well-formedness rules
- and (2) semantic alignment - accurate reflection of the intended meaning in
the source text. We present LADEX (LLM-based Activity Diagram Extractor), a
pipeline for deriving activity diagrams from natural-language process
descriptions using an LLM-driven critique-refine process. Structural checks in
LADEX can be performed either algorithmically or by an LLM, while alignment
checks are always performed by an LLM. We design five ablated variants of LADEX
to study: (i) the impact of the critique-refine loop itself, (ii) the role of
LLM-based semantic checks, and (iii) the comparative effectiveness of
algorithmic versus LLM-based structural checks.
  To evaluate LADEX, we compare the generated activity diagrams with
expert-created ground truths using trace-based operational semantics. This
enables automated measurement of correctness and completeness. Experiments on
two datasets indicate that: (1) the critique-refine loop improves structural
validity, correctness, and completeness compared to single-pass generation; (2)
algorithmic structural checks eliminate inconsistencies that LLM-based checks
fail to detect, improving correctness by an average of 17.81% and completeness
by 13.24% over LLM-only checks; and (3) combining algorithmic structural checks
with LLM-based semantic checks, implemented using the reasoning-focused O4
Mini, achieves the best overall performance - yielding average correctness of
up to 86.37% and average completeness of up to 88.56% - while requiring fewer
than five LLM calls on average.

### 2. [Are We SOLID Yet? An Empirical Study on Prompting LLMs to Detect Design Principle Violations](http://arxiv.org/pdf/2509.03093v1)

Authors: Fatih Pehlivan, Arçin Ülkü Ergüzen, Sahand Moslemi Yengejeh, Mayasah Lami, Anil Koyuncu

Traditional static analysis methods struggle to detect semantic design flaws,
such as violations of the SOLID principles, which require a strong
understanding of object-oriented design patterns and principles. Existing
solutions typically focus on individual SOLID principles or specific
programming languages, leaving a gap in the ability to detect violations across
all five principles in multi-language codebases. This paper presents a new
approach: a methodology that leverages tailored prompt engineering to assess
LLMs on their ability to detect SOLID violations across multiple languages. We
present a benchmark of four leading LLMs-CodeLlama, DeepSeekCoder, QwenCoder,
and GPT-4o Mini-on their ability to detect violations of all five SOLID
principles. For this evaluation, we construct a new benchmark dataset of 240
manually validated code examples. Using this dataset, we test four distinct
prompt strategies inspired by established zero-shot, few-shot, and
chain-of-thought techniques to systematically measure their impact on detection
accuracy. Our emerging results reveal a stark hierarchy among models, with
GPT-4o Mini decisively outperforming others, yet even struggles with
challenging principles like DIP. Crucially, we show that prompt strategy has a
dramatic impact, but no single strategy is universally best; for instance, a
deliberative ENSEMBLE prompt excels at OCP detection while a hint-based EXAMPLE
prompt is superior for DIP violations. Across all experiments, detection
accuracy is heavily influenced by language characteristics and degrades sharply
with increasing code complexity. These initial findings demonstrate that
effective, AI-driven design analysis requires not a single best model, but a
tailored approach that matches the right model and prompt to the specific
design context, highlighting the potential of LLMs to support maintainability
through AI-assisted code analysis.

### 3. [TopoMap: A Feature-based Semantic Discriminator of the Topographical Regions in the Test Input Space](http://arxiv.org/pdf/2509.03242v1)

Authors: Gianmarco De Vita, Nargiz Humbatova, Paolo Tonella

Testing Deep Learning (DL)-based systems is an open challenge. Although it is
relatively easy to find inputs that cause a DL model to misbehave, the grouping
of inputs by features that make the DL model under test fail is largely
unexplored. Existing approaches for DL testing introduce perturbations that may
focus on specific failure-inducing features, while neglecting others that
belong to different regions of the feature space. In this paper, we create an
explicit topographical map of the input feature space. Our approach, named
TopoMap, is both black-box and model-agnostic as it relies solely on features
that characterise the input space. To discriminate the inputs according to the
specific features they share, we first apply dimensionality reduction to obtain
input embeddings, which are then subjected to clustering. Each DL model might
require specific embedding computations and clustering algorithms to achieve a
meaningful separation of inputs into discriminative groups. We propose a novel
way to evaluate alternative configurations of embedding and clustering
techniques. We used a deep neural network (DNN) as an approximation of a human
evaluator who could tell whether a pair of clusters can be discriminated based
on the features of the included elements. We use such a DNN to automatically
select the optimal topographical map of the inputs among all those that are
produced by different embedding/clustering configurations. The evaluation
results show that the maps generated by TopoMap consist of distinguishable and
meaningful regions. In addition, we evaluate the effectiveness of TopoMap using
mutation analysis. In particular, we assess whether the clusters in our
topographical map allow for an effective selection of mutation-killing inputs.
Experimental results show that our approach outperforms random selection by 35%
on average on killable mutants; by 61% on non-killable ones.

### 4. [AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation](http://arxiv.org/pdf/2509.03270v1)

Authors: Martin Skoglund, Fredrik Warg, Aria Mirzai, Anders Thorsen, Karl Lundgren, Peter Folkesson, Bastian Havers-zulka

Integrating Artificial Intelligence (AI) technology in electric vehicles (EV)
introduces unique challenges for safety assurance, particularly within the
framework of ISO 26262, which governs functional safety in the automotive
domain. Traditional assessment methodologies are not geared toward evaluating
AI-based functions and require evolving standards and practices. This paper
explores how an independent assessment of an AI component in an EV can be
achieved when combining ISO 26262 with the recently released ISO/PAS 8800,
whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC)
battery estimation exemplifies the process. Key features relevant to the
independent assessment of this extended evaluation approach are identified. As
part of the evaluation, robustness testing of the AI component is conducted
using fault injection experiments, wherein perturbed sensor inputs are
systematically introduced to assess the component's resilience to input
variance.

### 5. [An experience-based classification of quantum bugs in quantum software](http://arxiv.org/pdf/2509.03280v1)

Authors: Nils Quetschlich, Olivia Di Matteo

As quantum computers continue to improve in quality and scale, there is a
growing need for accessible software frameworks for programming them. However,
the unique behavior of quantum systems means specialized approaches, beyond
traditional software development, are required. This is particularly true for
debugging due to quantum bugs, i.e., bugs that occur precisely because an
algorithm is a quantum algorithm. Pinpointing a quantum bug's root cause often
requires significant developer time, as there is little established guidance
for quantum debugging techniques. Developing such guidance is the main
challenge we sought to address. In this work, we describe a set of 14 quantum
bugs, sourced primarily from our experience as quantum software developers, and
supplemented by analysis of open-source GitHub repositories. We detail their
context, symptoms, and the techniques applied to identify and fix them. While
classifying these bugs based on existing schemes, we observed that most emerged
due to unique interactions between multiple aspects of an algorithm or
workflow. In other words, they occurred because more than one thing went wrong,
which provided important insight into why quantum debugging is more
challenging. Furthermore, based on this clustering, we found that -
unexpectedly - there is no clear relationship between debugging strategies and
bug classes. Further research is needed to develop effective and systematic
quantum debugging strategies.

### 6. [app.build: A Production Framework for Scaling Agentic Prompt-to-App Generation with Environment Scaffolding](http://arxiv.org/pdf/2509.03310v1)

Authors: Evgenii Kniazev, Arseny Kravchenko, Igor Rekun, James Broadhead, Nikita Shamgunov, Pranav Sah, Pratik Nichite, Ivan Yamshchikov

We present app.build (https://github.com/appdotbuild/agent/), an open-source
framework that improves LLM-based application generation through systematic
validation and structured environments. Our approach combines multi-layered
validation pipelines, stack-specific orchestration, and model-agnostic
architecture, implemented across three reference stacks. Through evaluation on
30 generation tasks, we demonstrate that comprehensive validation achieves
73.3% viability rate with 30% reaching perfect quality scores, while
open-weights models achieve 80.8% of closed-model performance when provided
structured environments. The open-source framework has been adopted by the
community, with over 3,000 applications generated to date. This work
demonstrates that scaling reliable AI agents requires scaling environments, not
just models -- providing empirical insights and complete reference
implementations for production-oriented agent systems.

### 7. [VulnRepairEval: An Exploit-Based Evaluation Framework for Assessing Large Language Model Vulnerability Repair Capabilities](http://arxiv.org/pdf/2509.03331v1)

Authors: Weizhe Wang, Wei Ma, Qiang Hu, Yao Zhang, Jianfei Sun, Bin Wu, Yang Liu, Guangquan Xu, Lingxiao Jiang

The adoption of Large Language Models (LLMs) for automated software
vulnerability patching has shown promising outcomes on carefully curated
evaluation sets. Nevertheless, existing datasets predominantly rely on
superficial validation methods rather than exploit-based verification, leading
to overestimated performance in security-sensitive applications. This paper
introduces VulnRepairEval, an evaluation framework anchored in functional
Proof-of-Concept (PoC) exploits. Our framework delivers a comprehensive,
containerized evaluation pipeline that enables reproducible differential
assessment, where repair success requires the original exploit to fail
execution against the modified code. The benchmark construction involved
extensive data curation: we processed over 400 CVEs and approximately 2,500
potential sources to extract a collection of authentic vulnerability instances
(23 Python CVEs) amenable to automated testing with working PoCs. Through
VulnRepairEval, we conduct a comprehensive evaluation of 12 popular LLMs and
observe a significant performance deficit: even the top-performing model
successfully addresses merely 5/23 instances (about 21.7%), exposing critical
weaknesses in security-focused applications. Our failure analysis reveals that
most unsuccessful attempts stem from imprecise vulnerability identification and
patches containing syntactic or semantic errors. Enhanced prompting strategies
and multi-agent approaches yield minimal improvements, with overall
effectiveness remaining largely unaffected. This work contributes a stringent,
practical evaluation framework for LLM-driven vulnerability remediation and
underscores the necessity for assessment protocols that authentically reflect
real-world exploitation scenarios.

### 8. [TraceLLM: Security Diagnosis Through Traces and Smart Contracts in Ethereum](http://arxiv.org/pdf/2509.03037v1)

Authors: Shuzheng Wang, Yue Huang, Zhuoer Xu, Yuming Huang, Jing Tang

Ethereum smart contracts hold tens of billions of USD in DeFi and NFTs, yet
comprehensive security analysis remains difficult due to unverified code,
proxy-based architectures, and the reliance on manual inspection of complex
execution traces. Existing approaches fall into two main categories: anomaly
transaction detection, which flags suspicious transactions but offers limited
insight into specific attack strategies hidden in execution traces inside
transactions, and code vulnerability detection, which cannot analyze unverified
contracts and struggles to show how identified flaws are exploited in real
incidents. As a result, analysts must still manually align transaction traces
with contract code to reconstruct attack scenarios and conduct forensics. To
address this gap, TraceLLM is proposed as a framework that leverages LLMs to
integrate execution trace-level detection with decompiled contract code. We
introduce a new anomaly execution path identification algorithm and an
LLM-refined decompile tool to identify vulnerable functions and provide
explicit attack paths to LLM. TraceLLM establishes the first benchmark for
joint trace and contract code-driven security analysis. For comparison, proxy
baselines are created by jointly transmitting the results of three
representative code analysis along with raw traces to LLM. TraceLLM identifies
attacker and victim addresses with 85.19\% precision and produces automated
reports with 70.37\% factual precision across 27 cases with ground truth expert
reports, achieving 25.93\% higher accuracy than the best baseline. Moreover,
across 148 real-world Ethereum incidents, TraceLLM automatically generates
reports with 66.22\% expert-verified accuracy, demonstrating strong
generalizability.

### Social and Information Networks

### 1. [Temporal social network modeling of mobile connectivity data with graph neural networks](http://arxiv.org/pdf/2509.03319v1)

Authors: Joel Jaskari, Chandreyee Roy, Fumiko Ogushi, Mikko Saukkoriipi, Jaakko Sahlsten, Kimmo Kaski

Graph neural networks (GNNs) have emerged as a state-of-the-art data-driven
tool for modeling connectivity data of graph-structured complex networks and
integrating information of their nodes and edges in space and time. However, as
of yet, the analysis of social networks using the time series of people's
mobile connectivity data has not been extensively investigated. In the present
study, we investigate four snapshot - based temporal GNNs in predicting the
phone call and SMS activity between users of a mobile communication network. In
addition, we develop a simple non - GNN baseline model using recently proposed
EdgeBank method. Our analysis shows that the ROLAND temporal GNN outperforms
the baseline model in most cases, whereas the other three GNNs perform on
average worse than the baseline. The results show that GNN based approaches
hold promise in the analysis of temporal social networks through mobile
connectivity data. However, due to the relatively small performance margin
between ROLAND and the baseline model, further research is required on
specialized GNN architectures for temporal social network analysis.

### 2. [VQualA 2025 Challenge on Engagement Prediction for Short Videos: Methods and Results](http://arxiv.org/pdf/2509.02969v1)

Authors: Dasong Li, Sizhuo Ma, Hang Hua, Wenjie Li, Jian Wang, Chris Wei Zhou, Fengbin Guan, Xin Li, Zihao Yu, Yiting Lu, Ru-Ling Liao, Yan Ye, Zhibo Chen, Wei Sun, Linhan Cao, Yuqin Cao, Weixia Zhang, Wen Wen, Kaiwei Zhang, Zijian Chen, Fangfang Lu, Xiongkuo Min, Guangtao Zhai, Erjia Xiao, Lingfeng Zhang, Zhenjie Su, Hao Cheng, Yu Liu, Renjing Xu, Long Chen, Xiaoshuai Hao, Zhenpeng Zeng, Jianqin Wu, Xuxu Wang, Qian Yu, Bo Hu, Weiwei Wang, Pinxin Liu, Yunlong Tang, Luchuan Song, Jinxi He, Jiaru Wu, Hanjia Lyu

This paper presents an overview of the VQualA 2025 Challenge on Engagement
Prediction for Short Videos, held in conjunction with ICCV 2025. The challenge
focuses on understanding and modeling the popularity of user-generated content
(UGC) short videos on social media platforms. To support this goal, the
challenge uses a new short-form UGC dataset featuring engagement metrics
derived from real-world user interactions. This objective of the Challenge is
to promote robust modeling strategies that capture the complex factors
influencing user engagement. Participants explored a variety of multi-modal
features, including visual content, audio, and metadata provided by creators.
The challenge attracted 97 participants and received 15 valid test submissions,
contributing significantly to progress in short-form UGC video engagement
prediction.

### Systems and Control

### 1. [Deep Reinforcement Learning-Based Decision-Making Strategy Considering User Satisfaction Feedback in Demand Response Program](http://arxiv.org/pdf/2509.02946v1)

Authors: Xin Li, Li Ding, Qiao Lin, Zhen-Wei Yu

Demand response providers (DRPs) are intermediaries between the upper-level
distribution system operator and the lower-level participants in demand
response (DR) programs. Usually, DRPs act as leaders and determine electricity
pricing strategies to maximize their economic revenue, while end-users adjust
their power consumption following the pricing signals. However, this
profit-seeking bi-level optimization model often neglects the satisfaction of
end-users participating in DR programs. In addition, the detailed mathematical
models underlying user decision-making strategy and satisfaction evaluation
mechanism are typically unavailable to DRPs, posing significant challenges to
conventional model-based solution methods. To address these issues, this paper
designs a user-side satisfaction evaluation mechanism and proposes a
multi-branch temporal fusion twin-delayed deep deterministic policy gradient
(MBTF-TD3) reinforcement learning algorithm. User satisfaction feedback is
incorporated into the reward function via a dynamically adjusted penalty term.
The proposed MBTF structure effectively extracts temporal feature dependencies
in the time-series observation data, and the dynamically adjusted penalty
function successfully enhances the overall satisfaction level of users. Several
experiments are conducted to validate the performance and the effectiveness of
our proposed solution algorithm.

### 2. [Spiking control systems for soft robotics: a rhythmic case study in a soft robotic crawler](http://arxiv.org/pdf/2509.02968v1)

Authors: Juncal Arbelaiz, Alessio Franci, Naomi Ehrich Leonard, Rodolphe Sepulchre, Bassam Bamieh

Inspired by spiking neural feedback, we propose a spiking controller for
efficient locomotion in a soft robotic crawler. Its bistability, akin to neural
fast positive feedback, combined with a sensorimotor slow negative feedback
loop, generates rhythmic spiking. The closed-loop system is robust through the
quantized actuation, and negative feedback ensures efficient locomotion with
minimal external tuning. We prove that peristaltic waves arise from a
supercritical Hopf bifurcation controlled by the sensorimotor gain. Dimensional
analysis reveals a separation of mechanical and electrical timescales, and
Geometric Singular Perturbation analysis explains endogenous crawling through
relaxation oscillations. We further formulate and analytically solve an
optimization problem in the singularly perturbed regime, proving that crawling
at mechanical resonance maximizes speed by a matching of neuromechanical
scales. Given the importance and ubiquity of rhythms and waves in soft-bodied
locomotion, we envision that spiking control systems could be utilized in a
variety of soft-robotic morphologies and modular distributed architectures,
yielding significant robustness, adaptability, and energetic gains across
scales.

### 3. [Target Enclosing Control for Nonholonomic Multi-Agent Systems with Connectivity Maintenance and Collision Avoidance](http://arxiv.org/pdf/2509.03168v1)

Authors: Boyin Zheng, Yahui Hao, Lu Liu

This article addresses the moving target enclosing control problem for
nonholonomic multi-agent systems with guaranteed network connectivity and
collision avoidance. We propose a novel control scheme to handle distance
constraints imposed by the agents' limited interaction ranges and
collision-free thresholds. By leveraging a Henneberg construction method, we
innovatively formulate the target enclosing requirements within an isostatic
distance-based formation framework, facilitating the integration of distance
constraints. Compared with existing results, our approach ensures the positive
definiteness of the underlying rigidity matrix and does not require controlling
the target's motion. To eliminate the occurrences of control singularities
caused by nonholonomic constraints, we propose a fixed-time angular control law
using barrier Lyapunov functions. Additionally, we develop a linear velocity
control law using the prescribed performance control approach and transformed
error constraints. We rigorously prove that our control laws enable the
multi-agent system to asymptotically achieve the desired angular formation
pattern around a moving target while satisfying the established distance
constraints. Finally, a simulation example is provided to validate the
effectiveness of the proposed method.

### 4. [Globally Asymptotically Stable Trajectory Tracking of Underactuated UAVs using Geometric Algebra](http://arxiv.org/pdf/2509.03484v1)

Authors: Ignacio Rubio Scola, Omar Alejandro Garcia Alcantara, Steven Sandoval, Eduardo Steed Espinoza Quesada, Hernan Haimovich, Luis Rodolfo Garcia Carrillo

This paper employs Geometric Algebra (GA) tools to model the dynamics of
objects in 3-dimensional space, serving as a proof of concept to facilitate
control design for trajectory tracking in underactuated systems. For control
purposes, the model is structured as a cascade system, where a rotational
subsystem drives a translational one. The rotational subsystem is linear, while
the translational subsystem follows a linear-plus-perturbation form, thereby
reducing the complexity of control design. A control strategy requiring only
simple operations, no memory, and no iterative search loops is presented to
illustrate the main features of the GA model. By employing GA to model both
translations and rotations, a singularity-free and geometrically intuitive
representation can be achieved through the use of the geometric product.
Closed-loop stability is rigorously established using input-to-state stability
methods. Numerical simulations of a quad tilt-rotorcraft performing trajectory
tracking in a windy environment validate the controller's stability and
performance.

### 5. [Multi-layer Digital Twin System for Future Mobile Metaverse](http://arxiv.org/pdf/2509.03049v1)

Authors: Gaosheng Zhao, Dong In Kim

In the upcoming 6G era, the communication networks are expected to face
unprecedented challenges in terms of complexity and dynamics. Digital Twin (DT)
technology, with its various digital capabilities, holds great potential to
facilitate the transformation of the communication network from passive
responding to proactive adaptation. Thus, in this paper, we propose a
multi-layer DT system that coordinates local DT, edge DT, and cloud DT for
future network architecture and functions. In our vision, the proposed DT
system will not only achieve real-time data-driven decision-making and digital
agent functions previously handled by centralized DT, but will do so in a more
distributed, mobile, layer-by-layer manner. Moreover, it will supply essential
data, pre-trained models, and open interfaces for future metaverse
applications, enabling creators and users to efficiently develop and experience
metaverse services.

### 6. [Forbal: Force Balanced 2-5 Degree of Freedom Robot Manipulator Built from a Five Bar Linkage](http://arxiv.org/pdf/2509.03119v1)

Authors: Yash Vyas, Matteo Bottin

A force balanced manipulator design based on the closed chain planar five bar
linkage is developed and experimentally validated. We present 2 variants as a
modular design: Forbal-2, a planar 2-DOF manipulator, and its extension to
5-DOF spatial motion called Forbal-5. The design considerations in terms of
geometric, kinematic, and dynamic design that fulfill the force balance
conditions while maximizing workspace are discussed. Then, the inverse
kinematics of both variants are derived from geometric principles.
  We validate the improvements from force balancing the manipulator through
comparative experiments with counter mass balanced and unbalanced
configurations. The results show how the balanced configuration yields a
reduction in the average reaction moments of up to 66\%, a reduction of average
joint torques of up to 79\%, as well as a noticeable reduction in position
error for Forbal-2. For Forbal-5, which has a higher end effector payload mass,
the joint torques are reduced up to 84\% for the balanced configuration.
Experimental results validate that the balanced manipulator design is suitable
for applications where the reduction of joint torques and reaction
forces/moments helps achieve millimeter level precision.

### 7. [Vibration Damping in Underactuated Cable-suspended Artwork -- Flying Belt Motion Control](http://arxiv.org/pdf/2509.03238v1)

Authors: Martin Goubej, Lauria Clarke, Martin Hrabačka, David Tolar

This paper presents a comprehensive refurbishment of the interactive robotic
art installation Standards and Double Standards by Rafael Lozano-Hemmer. The
installation features an array of belts suspended from the ceiling, each
actuated by stepper motors and dynamically oriented by a vision-based tracking
system that follows the movements of exhibition visitors. The original system
was limited by oscillatory dynamics, resulting in torsional and pendulum-like
vibrations that constrained rotational speed and reduced interactive
responsiveness. To address these challenges, the refurbishment involved
significant upgrades to both hardware and motion control algorithms. A detailed
mathematical model of the flying belt system was developed to accurately
capture its dynamic behavior, providing a foundation for advanced control
design. An input shaping method, formulated as a convex optimization problem,
was implemented to effectively suppress vibrations, enabling smoother and
faster belt movements. Experimental results demonstrate substantial
improvements in system performance and audience interaction. This work
exemplifies the integration of robotics, control engineering, and interactive
art, offering new solutions to technical challenges in real-time motion control
and vibration damping for large-scale kinetic installations.

### 8. [Hidden Convexity in Active Learning: A Convexified Online Input Design for ARX Systems](http://arxiv.org/pdf/2509.03257v1)

Authors: Nicolas Chatzikiriakos, Bowen Song, Philipp Rank, Andrea Iannelli

The goal of this work is to accelerate the identification of an unknown ARX
system from trajectory data through online input design. Specifically, we
present an active learning algorithm that sequentially selects the input to
excite the system according to an experiment design criterion using the past
measured data. The adopted criterion yields a non-convex optimization problem,
but we provide an exact convex reformulation allowing to find the global
optimizer in a computationally tractable way. Moreover, we give sample
complexity bounds on the estimation error due to the stochastic noise.
Numerical studies showcase the effectiveness of our algorithm and the benefits
of the convex reformulation.

### 9. [Parallel-Constraint Model Predictive Control: Exploiting Parallel Computation for Improving Safety](http://arxiv.org/pdf/2509.03261v1)

Authors: Elias Fontanari, Gianni Lunardi, Matteo Saveriano, Andrea Del Prete

Ensuring constraint satisfaction is a key requirement for safety-critical
systems, which include most robotic platforms. For example, constraints can be
used for modeling joint position/velocity/torque limits and collision
avoidance. Constrained systems are often controlled using Model Predictive
Control, because of its ability to naturally handle constraints, relying on
numerical optimization. However, ensuring constraint satisfaction is
challenging for nonlinear systems/constraints. A well-known tool to make
controllers safe is the so-called control-invariant set (a.k.a. safe set). In
our previous work, we have shown that safety can be improved by letting the
safe-set constraint recede along the MPC horizon. In this paper, we push that
idea further by exploiting parallel computation to improve safety. We solve
several MPC problems at the same time, where each problem instantiates the
safe-set constraint at a different time step along the horizon. Finally, the
controller can select the best solution according to some user-defined
criteria. We validated this idea through extensive simulations with a 3-joint
robotic arm, showing that significant improvements can be achieved in terms of
safety and performance, even using as little as 4 computational cores.

### 10. [Machine Learning-Driven Anomaly Detection for 5G O-RAN Performance Metrics](http://arxiv.org/pdf/2509.03290v1)

Authors: Babak Azkaei, Kishor Chandra Joshi, George Exarchakos

The ever-increasing reliance of critical services on network infrastructure
coupled with the increased operational complexity of beyond-5G/6G networks
necessitate the need for proactive and automated network fault management. The
provision for open interfaces among different radio access network\,(RAN)
elements and the integration of AI/ML into network architecture enabled by the
Open RAN\,(O-RAN) specifications bring new possibilities for active network
health monitoring and anomaly detection. In this paper we leverage these
advantages and develop an anomaly detection framework that proactively detect
the possible throughput drops for a UE and minimize the post-handover failures.
We propose two actionable anomaly detection algorithms tailored for real-world
deployment. The first algorithm identifies user equipment (UE) at risk of
severe throughput degradation by analyzing key performance indicators (KPIs)
such as resource block utilization and signal quality metrics, enabling
proactive handover initiation. The second algorithm evaluates neighbor cell
radio coverage quality, filtering out cells with anomalous signal strength or
interference levels. This reduces candidate targets for handover by 41.27\% on
average. Together, these methods mitigate post-handover failures and throughput
drops while operating much faster than the near-real-time latency constraints.
This paves the way for self-healing 6G networks.

### Machine Learning (Statistics Category)

### 1. [LSAM: Asynchronous Distributed Training with Landscape-Smoothed Sharpness-Aware Minimization](http://arxiv.org/pdf/2509.03110v1)

Authors: Yunfei Teng, Sixin Zhang

While Sharpness-Aware Minimization (SAM) improves generalization in deep
neural networks by minimizing both loss and sharpness, it suffers from
inefficiency in distributed large-batch training. We present Landscape-Smoothed
SAM (LSAM), a novel optimizer that preserves SAM's generalization advantages
while offering superior efficiency. LSAM integrates SAM's adversarial steps
with an asynchronous distributed sampling strategy, generating an asynchronous
distributed sampling scheme, producing a smoothed sharpness-aware loss
landscape for optimization. This design eliminates synchronization bottlenecks,
accelerates large-batch convergence, and delivers higher final accuracy
compared to data-parallel SAM.

### 2. [Feedback-Enhanced Online Multiple Testing with Applications to Conformal Selection](http://arxiv.org/pdf/2509.03297v1)

Authors: Lin Lu, Yuyang Huo, Haojie Ren, Zhaojun Wang, Changliang Zou

We study online multiple testing with feedback, where decisions are made
sequentially and the true state of the hypothesis is revealed after the
decision has been made, either instantly or with a delay. We propose GAIF, a
feedback-enhanced generalized alpha-investing framework that dynamically
adjusts thresholds using revealed outcomes, ensuring finite-sample false
discovery rate (FDR)/marginal FDR control. Extending GAIF to online conformal
testing, we construct independent conformal $p$-values and introduce a
feedback-driven model selection criterion to identify the best model/score,
thereby improving statistical power. We demonstrate the effectiveness of our
methods through numerical simulations and real-data applications.

### 3. [Bayesian Additive Regression Trees for functional ANOVA model](http://arxiv.org/pdf/2509.03317v1)

Authors: Seokhun Park, Insung Kong, Yongdai Kim

Bayesian Additive Regression Trees (BART) is a powerful statistical model
that leverages the strengths of Bayesian inference and regression trees. It has
received significant attention for capturing complex non-linear relationships
and interactions among predictors. However, the accuracy of BART often comes at
the cost of interpretability. To address this limitation, we propose ANOVA
Bayesian Additive Regression Trees (ANOVA-BART), a novel extension of BART
based on the functional ANOVA decomposition, which is used to decompose the
variability of a function into different interactions, each representing the
contribution of a different set of covariates or factors. Our proposed
ANOVA-BART enhances interpretability, preserves and extends the theoretical
guarantees of BART, and achieves superior predictive performance. Specifically,
we establish that the posterior concentration rate of ANOVA-BART is nearly
minimax optimal, and further provides the same convergence rates for each
interaction that are not available for BART. Moreover, comprehensive
experiments confirm that ANOVA-BART surpasses BART in both accuracy and
uncertainty quantification, while also demonstrating its effectiveness in
component selection. These results suggest that ANOVA-BART offers a compelling
alternative to BART by balancing predictive accuracy, interpretability, and
theoretical consistency.

### 4. [The distribution of calibrated likelihood functions on the probability-likelihood Aitchison simplex](http://arxiv.org/pdf/2509.03365v1)

Authors: Paul-Gauthier Noé, Andreas Nautsch, Driss Matrouf, Pierre-Michel Bousquet, Jean-François Bonastre

While calibration of probabilistic predictions has been widely studied, this
paper rather addresses calibration of likelihood functions. This has been
discussed, especially in biometrics, in cases with only two exhaustive and
mutually exclusive hypotheses (classes) where likelihood functions can be
written as log-likelihood-ratios (LLRs). After defining calibration for LLRs
and its connection with the concept of weight-of-evidence, we present the
idempotence property and its associated constraint on the distribution of the
LLRs. Although these results have been known for decades, they have been
limited to the binary case. Here, we extend them to cases with more than two
hypotheses by using the Aitchison geometry of the simplex, which allows us to
recover, in a vector form, the additive form of the Bayes' rule; extending
therefore the LLR and the weight-of-evidence to any number of hypotheses.
Especially, we extend the definition of calibration, the idempotence, and the
constraint on the distribution of likelihood functions to this multiple
hypotheses and multiclass counterpart of the LLR: the isometric-log-ratio
transformed likelihood function. This work is mainly conceptual, but we still
provide one application to machine learning by presenting a non-linear
discriminant analysis where the discriminant components form a calibrated
likelihood function over the classes, improving therefore the interpretability
and the reliability of the method.

### 5. [Understanding and Improving the Shampoo Optimizer via Kullback-Leibler Minimization](http://arxiv.org/pdf/2509.03378v1)

Authors: Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, Roger B. Grosse

As an adaptive method, Shampoo employs a structured second-moment estimation,
and its effectiveness has attracted growing attention. Prior work has primarily
analyzed its estimation scheme through the Frobenius norm. Motivated by the
natural connection between the second moment and a covariance matrix, we
propose studying Shampoo's estimation as covariance estimation through the lens
of Kullback-Leibler (KL) minimization. This alternative perspective reveals a
previously hidden limitation, motivating improvements to Shampoo's design.
Building on this insight, we develop a practical estimation scheme, termed
KL-Shampoo, that eliminates Shampoo's reliance on Adam for stabilization,
thereby removing the additional memory overhead introduced by Adam. Preliminary
results show that KL-Shampoo improves Shampoo's performance, enabling it to
stabilize without Adam and even outperform its Adam-stabilized variant, SOAP,
in neural network pretraining.

### 6. [Non-Linear Counterfactual Aggregate Optimization](http://arxiv.org/pdf/2509.03438v1)

Authors: Benjamin Heymann, Otmane Sakhi

We consider the problem of directly optimizing a non-linear function of an
outcome, where this outcome itself is the sum of many small contributions. The
non-linearity of the function means that the problem is not equivalent to the
maximization of the expectation of the individual contribution. By leveraging
the concentration properties of the sum of individual outcomes, we derive a
scalable descent algorithm that directly optimizes for our stated objective.
This allows for instance to maximize the probability of successful A/B test,
for which it can be wiser to target a success criterion, such as exceeding a
given uplift, rather than chasing the highest expected payoff.

### 7. [Off-Policy Learning in Large Action Spaces: Optimization Matters More Than Estimation](http://arxiv.org/pdf/2509.03456v1)

Authors: Imad Aouali, Otmane Sakhi

Off-policy evaluation (OPE) and off-policy learning (OPL) are foundational
for decision-making in offline contextual bandits. Recent advances in OPL
primarily optimize OPE estimators with improved statistical properties,
assuming that better estimators inherently yield superior policies. Although
theoretically justified, we argue this estimator-centric approach neglects a
critical practical obstacle: challenging optimization landscapes. In this
paper, we provide theoretical insights and extensive empirical evidence showing
that current OPL methods encounter severe optimization issues, particularly as
action spaces become large. We demonstrate that simpler weighted log-likelihood
objectives enjoy substantially better optimization properties and still recover
competitive, often superior, learned policies. Our findings emphasize the
necessity of explicitly addressing optimization considerations in the
development of OPL algorithms for large action spaces.

### 8. [Faster Gradient Methods for Highly-smooth Stochastic Bilevel Optimization](http://arxiv.org/pdf/2509.02937v1)

Authors: Lesi Chen, Junru Li, Jingzhao Zhang

This paper studies the complexity of finding an $\epsilon$-stationary point
for stochastic bilevel optimization when the upper-level problem is nonconvex
and the lower-level problem is strongly convex. Recent work proposed the
first-order method, F${}^2$SA, achieving the
$\tilde{\mathcal{O}}(\epsilon^{-6})$ upper complexity bound for first-order
smooth problems. This is slower than the optimal $\Omega(\epsilon^{-4})$
complexity lower bound in its single-level counterpart. In this work, we show
that faster rates are achievable for higher-order smooth problems. We first
reformulate F$^2$SA as approximating the hyper-gradient with a forward
difference. Based on this observation, we propose a class of methods
F${}^2$SA-$p$ that uses $p$th-order finite difference for hyper-gradient
approximation and improves the upper bound to $\tilde{\mathcal{O}}(p
\epsilon^{4-p/2})$ for $p$th-order smooth problems. Finally, we demonstrate
that the $\Omega(\epsilon^{-4})$ lower bound also holds for stochastic bilevel
problems when the high-order smoothness holds for the lower-level variable,
indicating that the upper bound of F${}^2$SA-$p$ is nearly optimal in the
highly smooth region $p = \Omega( \log \epsilon^{-1} / \log \log
\epsilon^{-1})$.

### 9. [Convergence for adaptive resampling of random Fourier features](http://arxiv.org/pdf/2509.03151v1)

Authors: Xin Huang, Aku Kammonen, Anamika Pandey, Mattias Sandberg, Erik von Schwerin, Anders Szepessy, Raúl Tempone

The machine learning random Fourier feature method for data in high dimension
is computationally and theoretically attractive since the optimization is based
on a convex standard least squares problem and independent sampling of Fourier
frequencies. The challenge is to sample the Fourier frequencies well. This work
proves convergence of a data adaptive method based on resampling the
frequencies asymptotically optimally, as the number of nodes and amount of data
tend to infinity. Numerical results based on resampling and adaptive random
walk steps together with approximations of the least squares problem by
conjugate gradient iterations confirm the analysis for regression and
classification problems.

### 10. [Markov Missing Graph: A Graphical Approach for Missing Data Imputation](http://arxiv.org/pdf/2509.03410v1)

Authors: Yanjiao Yang, Yen-Chi Chen

We introduce the Markov missing graph (MMG), a novel framework that imputes
missing data based on undirected graphs. MMG leverages conditional independence
relationships to locally decompose the imputation model. To establish the
identification, we introduce the Principle of Available Information (PAI),
which guides the use of all relevant observed data. We then propose a flexible
statistical learning paradigm, MMG Imputation Risk Minimization under PAI, that
frames the imputation task as an empirical risk minimization problem. This
framework is adaptable to various modeling choices. We develop theories of MMG,
including the connection between MMG and Little's complete-case missing value
assumption, recovery under missing completely at random, efficiency theory, and
graph-related properties. We show the validity of our method with simulation
studies and illustrate its application with a real-world Alzheimer's data set.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-04 PST.

### 1. [Reimagining clinical AI: from clickstreams to clinical insights with EHR use metadata](https://www.nature.com/articles/s44401-025-00040-5)

Authors: Chao Yan et al.

