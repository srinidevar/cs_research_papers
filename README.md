# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-09 17:11:42.827231 PST.

### Artificial Intelligence

### 1. [Belief Filtering for Epistemic Control in Linguistic State Space](http://arxiv.org/pdf/2505.04927v1)

Authors: Sebastian Dumbrava

We examine belief filtering as a mechanism for the epistemic control of
artificial agents, focusing on the regulation of internal cognitive states
represented as linguistic expressions. This mechanism is developed within the
Semantic Manifold framework, where belief states are dynamic, structured
ensembles of natural language fragments. Belief filters act as content-aware
operations on these fragments across various cognitive transitions. This paper
illustrates how the inherent interpretability and modularity of such a
linguistically-grounded cognitive architecture directly enable belief
filtering, offering a principled approach to agent regulation. The study
highlights the potential for enhancing AI safety and alignment through
structured interventions in an agent's internal semantic space and points to
new directions for architecturally embedded cognitive governance.

### 2. [Position: Epistemic Artificial Intelligence is Essential for Machine Learning Models to Know When They Do Not Know](http://arxiv.org/pdf/2505.04950v1)

Authors: Shireen Kudukkil Manchingal, Fabio Cuzzolin

Despite the impressive achievements of AI, including advancements in
generative models and large language models, there remains a significant gap in
the ability of AI to handle uncertainty and generalize beyond the training
data. We argue that AI models, especially in autonomous systems, fail to make
robust predictions when faced with unfamiliar or adversarial data, as evidenced
by incidents with autonomous vehicles. Traditional machine learning approaches
struggle to address these issues due to an overemphasis on data fitting and
domain adaptation. This position paper posits a paradigm shift towards
epistemic artificial intelligence, emphasizing the need for models to learn not
only from what they know but also from their ignorance. This approach, which
focuses on recognizing and managing uncertainty, offers a potential solution to
improve the resilience and robustness of AI systems, ensuring that they can
better handle unpredictable real-world environments.

### 3. [A Neuro-Symbolic Framework for Sequence Classification with Relational and Temporal Knowledge](http://arxiv.org/pdf/2505.05106v1)

Authors: Luca Salvatore Lorello, Marco Lippi, Stefano Melacci

One of the goals of neuro-symbolic artificial intelligence is to exploit
background knowledge to improve the performance of learning tasks. However,
most of the existing frameworks focus on the simplified scenario where
knowledge does not change over time and does not cover the temporal dimension.
In this work we consider the much more challenging problem of knowledge-driven
sequence classification where different portions of knowledge must be employed
at different timesteps, and temporal relations are available. Our experimental
evaluation compares multi-stage neuro-symbolic and neural-only architectures,
and it is conducted on a newly-introduced benchmarking framework. Results
demonstrate the challenging nature of this novel setting, and also highlight
under-explored shortcomings of neuro-symbolic methods, representing a precious
reference for future research.

### 4. [Is there a half-life for the success rates of AI agents?](http://arxiv.org/pdf/2505.05115v1)

Authors: Toby Ord

Building on the recent empirical work of Kwa et al. (2025), I show that
within their suite of research-engineering tasks the performance of AI agents
on longer-duration tasks can be explained by an extremely simple mathematical
model -- a constant rate of failing during each minute a human would take to do
the task. This implies an exponentially declining success rate with the length
of the task and that each agent could be characterised by its own half-life.
This empirical regularity allows us to estimate the success rate for an agent
at different task lengths. And the fact that this model is a good fit for the
data is suggestive of the underlying causes of failure on longer tasks -- that
they involve increasingly large sets of subtasks where failing any one fails
the task. Whether this model applies more generally on other suites of tasks is
unknown and an important subject for further work.

### 5. [ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints](http://arxiv.org/pdf/2505.05232v1)

Authors: Mahmoud Amiri, Thomas Bocklitz

The rapid expansion of chemistry literature poses significant challenges for
researchers seeking to efficiently access domain-specific knowledge. To support
advancements in chemistry-focused natural language processing (NLP), we present
ChemRxivQuest, a curated dataset of 970 high-quality question-answer (QA) pairs
derived from 155 ChemRxiv preprints across 17 subfields of chemistry. Each QA
pair is explicitly linked to its source text segment to ensure traceability and
contextual accuracy. ChemRxivQuest was constructed using an automated pipeline
that combines optical character recognition (OCR), GPT-4o-based QA generation,
and a fuzzy matching technique for answer verification. The dataset emphasizes
conceptual, mechanistic, applied, and experimental questions, enabling
applications in retrieval-based QA systems, search engine development, and
fine-tuning of domain-adapted large language models. We analyze the dataset's
structure, coverage, and limitations, and outline future directions for
expansion and expert validation. ChemRxivQuest provides a foundational resource
for chemistry NLP research, education, and tool development.

### 6. [Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation](http://arxiv.org/pdf/2505.05235v1)

Authors: Luca Marzari, Isabella Mastroeni, Alessandro Farinelli

Traditional methods for formal verification (FV) of deep neural networks
(DNNs) are constrained by a binary encoding of safety properties, where a model
is classified as either safe or unsafe (robust or not robust). This binary
encoding fails to capture the nuanced safety levels within a model, often
resulting in either overly restrictive or too permissive requirements. In this
paper, we introduce a novel problem formulation called Abstract
DNN-Verification, which verifies a hierarchical structure of unsafe outputs,
providing a more granular analysis of the safety aspect for a given DNN.
Crucially, by leveraging abstract interpretation and reasoning about output
reachable sets, our approach enables assessing multiple safety levels during
the FV process, requiring the same (in the worst case) or even potentially less
computational effort than the traditional binary verification approach.
Specifically, we demonstrate how this formulation allows rank adversarial
inputs according to their abstract safety level violation, offering a more
detailed evaluation of the model's safety and robustness. Our contributions
include a theoretical exploration of the relationship between our novel
abstract safety formulation and existing approaches that employ abstract
interpretation for robustness verification, complexity analysis of the novel
problem introduced, and an empirical evaluation considering both a complex deep
reinforcement learning task (based on Habitat 3.0) and standard
DNN-Verification benchmarks.

### 7. [EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation](http://arxiv.org/pdf/2505.05440v1)

Authors: Biao Yi, Xavier Hu, Yurun Chen, Shengyu Zhang, Hongxia Yang, Fan Wu, Fei Wu

Cloud-based mobile agents powered by (multimodal) large language models
((M)LLMs) offer strong reasoning abilities but suffer from high latency and
cost. While fine-tuned (M)SLMs enable edge deployment, they often lose general
capabilities and struggle with complex tasks. To address this, we propose
EcoAgent, an Edge-Cloud cOllaborative multi-agent framework for mobile
automation. EcoAgent features a closed-loop collaboration among a cloud-based
Planning Agent and two edge-based agents: the Execution Agent for action
execution and the Observation Agent for verifying outcomes. The Observation
Agent uses a Pre-Understanding Module to compress screen images into concise
text, reducing token usage. In case of failure, the Planning Agent retrieves
screen history and replans via a Reflection Module. Experiments on AndroidWorld
show that EcoAgent maintains high task success rates while significantly
reducing MLLM token consumption, enabling efficient and practical mobile
automation.

### 8. [Conversational Process Model Redesign](http://arxiv.org/pdf/2505.05453v1)

Authors: Nataliia Klievtsova, Timotheus Kampik, Juergen Mangler, Stefanie Rinderle-Ma

With the recent success of large language models (LLMs), the idea of
AI-augmented Business Process Management systems is becoming more feasible. One
of their essential characteristics is the ability to be conversationally
actionable, allowing humans to interact with the LLM effectively to perform
crucial process life cycle tasks such as process model design and redesign.
However, most current research focuses on single-prompt execution and
evaluation of results, rather than on continuous interaction between the user
and the LLM. In this work, we aim to explore the feasibility of using LLMs to
empower domain experts in the creation and redesign of process models in an
iterative and effective way. The proposed conversational process model redesign
(CPD) approach receives as input a process model and a redesign request by the
user in natural language. Instead of just letting the LLM make changes, the LLM
is employed to (a) identify process change patterns from literature, (b)
re-phrase the change request to be aligned with an expected wording for the
identified pattern (i.e., the meaning), and then to (c) apply the meaning of
the change to the process model. This multi-step approach allows for
explainable and reproducible changes. In order to ensure the feasibility of the
CPD approach, and to find out how well the patterns from literature can be
handled by the LLM, we performed an extensive evaluation. The results show that
some patterns are hard to understand by LLMs and by users. Within the scope of
the study, we demonstrated that users need support to describe the changes
clearly. Overall the evaluation shows that the LLMs can handle most changes
well according to a set of completeness and correctness criteria.

### 9. [Auto-regressive transformation for image alignment](http://arxiv.org/pdf/2505.04864v1)

Authors: Kanggeon Lee, Soochahn Lee, Kyoung Mu Lee

Existing methods for image alignment struggle in cases involving
feature-sparse regions, extreme scale and field-of-view differences, and large
deformations, often resulting in suboptimal accuracy. Robustness to these
challenges improves through iterative refinement of the transformation field
while focusing on critical regions in multi-scale image representations. We
thus propose Auto-Regressive Transformation (ART), a novel method that
iteratively estimates the coarse-to-fine transformations within an
auto-regressive framework. Leveraging hierarchical multi-scale features, our
network refines the transformations using randomly sampled points at each
scale. By incorporating guidance from the cross-attention layer, the model
focuses on critical regions, ensuring accurate alignment even in challenging,
feature-limited conditions. Extensive experiments across diverse datasets
demonstrate that ART significantly outperforms state-of-the-art methods,
establishing it as a powerful new method for precise image alignment with broad
applicability.

### 10. [Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning](http://arxiv.org/pdf/2505.04877v1)

Authors: Lianbo Ma, Jianlun Ma, Yuee Zhou, Guoyang Xie, Qiang He, Zhichao Lu

Mixed Precision Quantization (MPQ) has become an essential technique for
optimizing neural network by determining the optimal bitwidth per layer.
Existing MPQ methods, however, face a major hurdle: they require a
computationally expensive search for quantization policies on large-scale
datasets. To resolve this issue, we introduce a novel approach that first
searches for quantization policies on small datasets and then generalizes them
to large-scale datasets. This approach simplifies the process, eliminating the
need for large-scale quantization fine-tuning and only necessitating model
weight adjustment. Our method is characterized by three key techniques:
sharpness-aware minimization for enhanced quantization generalization, implicit
gradient direction alignment to handle gradient conflicts among different
optimization objectives, and an adaptive perturbation radius to accelerate
optimization. Both theoretical analysis and experimental results validate our
approach. Using the CIFAR10 dataset (just 0.5\% the size of ImageNet training
data) for MPQ policy search, we achieved equivalent accuracy on ImageNet with a
significantly lower computational cost, while improving efficiency by up to
150% over the baselines.

### Hardware Architecture

### 1. [PUDTune: Multi-Level Charging for High-Precision Calibration in Processing-Using-DRAM](http://arxiv.org/pdf/2505.05266v1)

Authors: Tatsuya Kubo, Daichi Tokuda, Lei Qu, Ting Cao, Shinya Takamaeda-Yamazaki

Recently, practical analog in-memory computing has been realized using
unmodified commercial DRAM modules. The underlying Processing-Using-DRAM (PUD)
techniques enable high-throughput bitwise operations directly within DRAM
arrays. However, the presence of inherent error-prone columns hinders PUD's
practical adoption. While selectively using only error-free columns would
ensure reliability, this approach significantly reduces PUD's computational
throughput.
  This paper presents PUDTune, a novel high-precision calibration technique for
increasing the number of error-free columns in PUD. PUDTune compensates for
errors by applying pre-identified column-specific offsets to PUD operations. By
leveraging multi-level charge states of DRAM cells, PUDTune generates
fine-grained and wide-range offset variations despite the limited available
rows. Our experiments with DDR4 DRAM demonstrate that PUDTune increases the
number of error-free columns by 1.81$\times$ compared to conventional
implementations, improving addition and multiplication throughput by
1.88$\times$ and 1.89$\times$ respectively.

### Computational Complexity

### 1. [Gap-preserving reductions and RE-completeness of independent set games](http://arxiv.org/pdf/2505.05253v1)

Authors: Laura Mančinska, Pieter Spaas, Taro Spirig

In complexity theory, gap-preserving reductions play a crucial role in
studying hardness of approximation and in analyzing the relative complexity of
multiprover interactive proof systems. In the quantum setting, multiprover
interactive proof systems with entangled provers correspond to gapped promise
problems for nonlocal games, and the recent result MIP$^*$=RE (Ji et al.,
arXiv:2001.04383) shows that these are in general undecidable. However, the
relative complexity of problems within MIP$^*$ is still not well-understood, as
establishing gap-preserving reductions in the quantum setting presents new
challenges. In this paper, we introduce a framework to study such reductions
and use it to establish MIP$^*$-completeness of the gapped promise problem for
the natural class of independent set games. In such a game, the goal is to
determine whether a given graph contains an independent set of a specified
size. We construct families of independent set games with constant question
size for which the gapped promise problem is undecidable. In contrast, the same
problem is decidable in polynomial time in the classical setting. To carry out
our reduction, we establish a new stability theorem, which could be of
independent interest, allowing us to perturb families of almost PVMs to genuine
PVMs.

### Computational Engineering

### 1. [Thermoelastic Kirchhoff Plate: A Novel Model for Shot Peen Forming Metal Panels](http://arxiv.org/pdf/2505.05236v1)

Authors: Conor Rowan

A common technique used in factories to shape metal panels is shot peen
forming, where the panel is sprayed with a high-velocity stream of small steel
pellets called shot. The impacts between the hard steel shot and softer
aluminum panel cause localized plastic deformation, both improving the fatigue
properties of the material's surface and imparting a residual stress
distribution that results in bending. Thus, a torque is associated with the
through-thickness shot peen stress distribution. We conceptualize shot peen
forming as the application of spatially varying torques, which are modeled with
the input of applied temperatures. In this paper, we derive the bending
equations for a thermally loaded homogeneous Kirchhoff plate in order to
predict the effects of shot peen forming. A simple test is devised to extract
the value of an equivalent applied torque from the bending response of
uniformly shot peened plates, which circumvents the difficulty of accounting
for surface plasticity. This torque can be used as an input to a model which
predicts the shape of rectangular plates under more complicated shot peen
conditions. An experiment is designed and carried out which investigates the
agreement between the model and real shot peen operations. The effect of
uncertainty in the experiment is estimated with Monte Carlo methods.

### 2. [Advanced Stock Market Prediction Using Long Short-Term Memory Networks: A Comprehensive Deep Learning Framework](http://arxiv.org/pdf/2505.05325v1)

Authors: Rajneesh Chaudhary

Predicting stock market movements remains a persistent challenge due to the
inherently volatile, non-linear, and stochastic nature of financial time series
data. This paper introduces a deep learning-based framework employing Long
Short-Term Memory (LSTM) networks to forecast the closing stock prices of major
technology firms: Apple, Google, Microsoft, and Amazon, listed on NASDAQ.
Historical data was sourced from Yahoo Finance and processed using
normalization and feature engineering techniques. The proposed model achieves a
Mean Absolute Percentage Error (MAPE) of 2.72 on unseen test data,
significantly outperforming traditional models like ARIMA. To further enhance
predictive accuracy, sentiment scores were integrated using real-time news
articles and social media data, analyzed through the VADER sentiment analysis
tool. A web application was also developed to provide real-time visualizations
of stock price forecasts, offering practical utility for both individual and
institutional investors. This research demonstrates the strength of LSTM
networks in modeling complex financial sequences and presents a novel hybrid
approach combining time series modeling with sentiment analysis.

### 3. [Physics-informed solution reconstruction in elasticity and heat transfer using the explicit constraint force method](http://arxiv.org/pdf/2505.04875v1)

Authors: Conor Rowan, Kurt Maute, Alireza Doostan

One use case of ``physics-informed neural networks'' (PINNs) is solution
reconstruction, which aims to estimate the full-field state of a physical
system from sparse measurements. Parameterized governing equations of the
system are used in tandem with the measurements to regularize the regression
problem. However, in real-world solution reconstruction problems, the
parameterized governing equation may be inconsistent with the physical
phenomena that give rise to the measurement data. We show that due to assuming
consistency between the true and parameterized physics, PINNs-based approaches
may fail to satisfy three basic criteria of interpretability, robustness, and
data consistency. As we argue, these criteria ensure that (i) the quality of
the reconstruction can be assessed, (ii) the reconstruction does not depend
strongly on the choice of physics loss, and (iii) that in certain situations,
the physics parameters can be uniquely recovered. In the context of elasticity
and heat transfer, we demonstrate how standard formulations of the physics loss
and techniques for constraining the solution to respect the measurement data
lead to different ``constraint forces" -- which we define as additional source
terms arising from the constraints -- and that these constraint forces can
significantly influence the reconstructed solution. To avoid the potentially
substantial influence of the choice of physics loss and method of constraint
enforcement on the reconstructed solution, we propose the ``explicit constraint
force method'' (ECFM) to gain control of the source term introduced by the
constraint. We then show that by satisfying the criteria of interpretability,
robustness, and data consistency, this approach leads to more predictable and
customizable reconstructions from noisy measurement data, even when the
parameterization of the missing physics is inconsistent with the measured
system.

### Computation and Language

### 1. [Rethinking the Relationship between the Power Law and Hierarchical Structures](http://arxiv.org/pdf/2505.04984v1)

Authors: Kai Nakaishi, Ryo Yoshida, Kohei Kajikawa, Koji Hukushima, Yohei Oseki

Statistical analysis of corpora provides an approach to quantitatively
investigate natural languages. This approach has revealed that several power
laws consistently emerge across different corpora and languages, suggesting the
universal principles underlying languages. Particularly, the power-law decay of
correlation has been interpreted as evidence for underlying hierarchical
structures in syntax, semantics, and discourse. This perspective has also been
extended to child languages and animal signals. However, the argument
supporting this interpretation has not been empirically tested. To address this
problem, this study examines the validity of the argument for syntactic
structures. Specifically, we test whether the statistical properties of parse
trees align with the implicit assumptions in the argument. Using English
corpora, we analyze the mutual information, deviations from probabilistic
context-free grammars (PCFGs), and other properties in parse trees, as well as
in the PCFG that approximates these trees. Our results indicate that the
assumptions do not hold for syntactic structures and that it is difficult to
apply the proposed argument to child languages and animal signals, highlighting
the need to reconsider the relationship between the power law and hierarchical
structures.

### 2. [Latent Preference Coding: Aligning Large Language Models via Discrete Latent Codes](http://arxiv.org/pdf/2505.04993v1)

Authors: Zhuocheng Gong, Jian Guan, Wei Wu, Huishuai Zhang, Dongyan Zhao

Large language models (LLMs) have achieved remarkable success, yet aligning
their generations with human preferences remains a critical challenge. Existing
approaches to preference modeling often rely on an explicit or implicit reward
function, overlooking the intricate and multifaceted nature of human
preferences that may encompass conflicting factors across diverse tasks and
populations. To address this limitation, we introduce Latent Preference Coding
(LPC), a novel framework that models the implicit factors as well as their
combinations behind holistic preferences using discrete latent codes. LPC
seamlessly integrates with various offline alignment algorithms, automatically
inferring the underlying factors and their importance from data without relying
on pre-defined reward functions and hand-crafted combination weights. Extensive
experiments on multiple benchmarks demonstrate that LPC consistently improves
upon three alignment algorithms (DPO, SimPO, and IPO) using three base models
(Mistral-7B, Llama3-8B, and Llama3-8B-Instruct). Furthermore, deeper analysis
reveals that the learned latent codes effectively capture the differences in
the distribution of human preferences and significantly enhance the robustness
of alignment against noise in data. By providing a unified representation for
the multifarious preference factors, LPC paves the way towards developing more
robust and versatile alignment techniques for the responsible deployment of
powerful LLMs.

### 3. [Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](http://arxiv.org/pdf/2505.05017v1)

Authors: Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Jiang Zong, Hao Peng, Jianwei Yin

Pre-trained large language models (LLMs) are commonly fine-tuned to adapt to
downstream tasks. Since the majority of knowledge is acquired during
pre-training, attributing the predictions of fine-tuned LLMs to their
pre-training data may provide valuable insights. Influence functions have been
proposed as a means to explain model predictions based on training data.
However, existing approaches fail to compute ``multi-stage'' influence and lack
scalability to billion-scale LLMs.
  In this paper, we propose the multi-stage influence function to attribute the
downstream predictions of fine-tuned LLMs to pre-training data under the
full-parameter fine-tuning paradigm. To enhance the efficiency and practicality
of our multi-stage influence function, we leverage Eigenvalue-corrected
Kronecker-Factored (EK-FAC) parameterization for efficient approximation.
Empirical results validate the superior scalability of EK-FAC approximation and
the effectiveness of our multi-stage influence function. Additionally, case
studies on a real-world LLM, dolly-v2-3b, demonstrate its interpretive power,
with exemplars illustrating insights provided by multi-stage influence
estimates. Our code is public at
https://github.com/colored-dye/multi_stage_influence_function.

### 4. [Performance Evaluation of Large Language Models in Bangla Consumer Health Query Summarization](http://arxiv.org/pdf/2505.05070v1)

Authors: Ajwad Abrar, Farzana Tabassum, Sabbir Ahmed

Consumer Health Queries (CHQs) in Bengali (Bangla), a low-resource language,
often contain extraneous details, complicating efficient medical responses.
This study investigates the zero-shot performance of nine advanced large
language models (LLMs): GPT-3.5-Turbo, GPT-4, Claude-3.5-Sonnet,
Llama3-70b-Instruct, Mixtral-8x22b-Instruct, Gemini-1.5-Pro,
Qwen2-72b-Instruct, Gemma-2-27b, and Athene-70B, in summarizing Bangla CHQs.
Using the BanglaCHQ-Summ dataset comprising 2,350 annotated query-summary
pairs, we benchmarked these LLMs using ROUGE metrics against Bangla T5, a
fine-tuned state-of-the-art model. Mixtral-8x22b-Instruct emerged as the top
performing model in ROUGE-1 and ROUGE-L, while Bangla T5 excelled in ROUGE-2.
The results demonstrate that zero-shot LLMs can rival fine-tuned models,
achieving high-quality summaries even without task-specific training. This work
underscores the potential of LLMs in addressing challenges in low-resource
languages, providing scalable solutions for healthcare query summarization.

### 5. [Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](http://arxiv.org/pdf/2505.05084v1)

Authors: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

The rapid advancement of large language models has raised significant
concerns regarding their potential misuse by malicious actors. As a result,
developing effective detectors to mitigate these risks has become a critical
priority. However, most existing detection methods focus excessively on
detection accuracy, often neglecting the societal risks posed by high false
positive rates (FPRs). This paper addresses this issue by leveraging Conformal
Prediction (CP), which effectively constrains the upper bound of FPRs. While
directly applying CP constrains FPRs, it also leads to a significant reduction
in detection performance. To overcome this trade-off, this paper proposes a
Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal
Prediction (MCP), which both enforces the FPR constraint and improves detection
performance. This paper also introduces RealDet, a high-quality dataset that
spans a wide range of domains, ensuring realistic calibration and enabling
superior detection performance when combined with MCP. Empirical evaluations
demonstrate that MCP effectively constrains FPRs, significantly enhances
detection performance, and increases robustness against adversarial attacks
across multiple detectors and datasets.

### 6. [Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](http://arxiv.org/pdf/2505.05111v1)

Authors: Boyi Deng, Yu Wan, Yidan Zhang, Baosong Yang, Fuli Feng

The mechanisms behind multilingual capabilities in Large Language Models
(LLMs) have been examined using neuron-based or internal-activation-based
methods. However, these methods often face challenges such as superposition and
layer-wise activation variance, which limit their reliability. Sparse
Autoencoders (SAEs) offer a more nuanced analysis by decomposing the
activations of LLMs into sparse linear combination of SAE features. We
introduce a novel metric to assess the monolinguality of features obtained from
SAEs, discovering that some features are strongly related to specific
languages. Additionally, we show that ablating these SAE features only
significantly reduces abilities in one language of LLMs, leaving others almost
unaffected. Interestingly, we find some languages have multiple synergistic SAE
features, and ablating them together yields greater improvement than ablating
individually. Moreover, we leverage these SAE-derived language-specific
features to enhance steering vectors, achieving control over the language
generated by LLMs.

### 7. [A Benchmark Dataset and a Framework for Urdu Multimodal Named Entity Recognition](http://arxiv.org/pdf/2505.05148v1)

Authors: Hussain Ahmad, Qingyang Zeng, Jing Wan

The emergence of multimodal content, particularly text and images on social
media, has positioned Multimodal Named Entity Recognition (MNER) as an
increasingly important area of research within Natural Language Processing.
Despite progress in high-resource languages such as English, MNER remains
underexplored for low-resource languages like Urdu. The primary challenges
include the scarcity of annotated multimodal datasets and the lack of
standardized baselines. To address these challenges, we introduce the U-MNER
framework and release the Twitter2015-Urdu dataset, a pioneering resource for
Urdu MNER. Adapted from the widely used Twitter2015 dataset, it is annotated
with Urdu-specific grammar rules. We establish benchmark baselines by
evaluating both text-based and multimodal models on this dataset, providing
comparative analyses to support future research on Urdu MNER. The U-MNER
framework integrates textual and visual context using Urdu-BERT for text
embeddings and ResNet for visual feature extraction, with a Cross-Modal Fusion
Module to align and fuse information. Our model achieves state-of-the-art
performance on the Twitter2015-Urdu dataset, laying the groundwork for further
MNER research in low-resource languages.

### 8. [QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation](http://arxiv.org/pdf/2505.05225v1)

Authors: Mengze Hong, Wailing Ng, Di Jiang, Chen Jason Zhang

The rapid advancement of Chinese large language models (LLMs) underscores the
need for domain-specific evaluations to ensure reliable applications. However,
existing benchmarks often lack coverage in vertical domains and offer limited
insights into the Chinese working context. Leveraging qualification exams as a
unified framework for human expertise evaluation, we introduce QualBench, the
first multi-domain Chinese QA benchmark dedicated to localized assessment of
Chinese LLMs. The dataset includes over 17,000 questions across six vertical
domains, with data selections grounded in 24 Chinese qualifications to closely
align with national policies and working standards. Through comprehensive
evaluation, the Qwen2.5 model outperformed the more advanced GPT-4o, with
Chinese LLMs consistently surpassing non-Chinese models, highlighting the
importance of localized domain knowledge in meeting qualification requirements.
The best performance of 75.26% reveals the current gaps in domain coverage
within model capabilities. Furthermore, we present the failure of LLM
collaboration with crowdsourcing mechanisms and suggest the opportunities for
multi-domain RAG knowledge enhancement and vertical domain LLM training with
Federated Learning.

### 9. [ICon: In-Context Contribution for Automatic Data Selection](http://arxiv.org/pdf/2505.05327v1)

Authors: Yixin Yang, Qingxiu Dong, Linli Yao, Fangwei Zhu, Zhifang Sui

Data selection for instruction tuning is essential for improving the
performance of Large Language Models (LLMs) and reducing training cost.
However, existing automated selection methods either depend on computationally
expensive gradient-based measures or manually designed heuristics, which may
fail to fully exploit the intrinsic attributes of data. In this paper, we
propose In-context Learning for Contribution Measurement (ICon), a novel
gradient-free method that takes advantage of the implicit fine-tuning nature of
in-context learning (ICL) to measure sample contribution without gradient
computation or manual indicators engineering. ICon offers a computationally
efficient alternative to gradient-based methods and reduces human inductive
bias inherent in heuristic-based approaches. ICon comprises three components
and identifies high-contribution data by assessing performance shifts under
implicit learning through ICL. Extensive experiments on three LLMs across 12
benchmarks and 5 pairwise evaluation sets demonstrate the effectiveness of
ICon. Remarkably, on LLaMA3.1-8B, models trained on 15% of ICon-selected data
outperform full datasets by 5.42% points and exceed the best performance of
widely used selection methods by 2.06% points. We further analyze
high-contribution samples selected by ICon, which show both diverse tasks and
appropriate difficulty levels, rather than just the hardest ones.

### 10. [Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?](http://arxiv.org/pdf/2505.05406v1)

Authors: Valeria Pastorino, Nafise Sadat Moosavi

Framing in media critically shapes public perception by selectively
emphasizing some details while downplaying others. With the rise of large
language models in automated news and content creation, there is growing
concern that these systems may introduce or even amplify framing biases
compared to human authors. In this paper, we explore how framing manifests in
both out-of-the-box and fine-tuned LLM-generated news content. Our analysis
reveals that, particularly in politically and socially sensitive contexts, LLMs
tend to exhibit more pronounced framing than their human counterparts. In
addition, we observe significant variation in framing tendencies across
different model architectures, with some models displaying notably higher
biases. These findings point to the need for effective post-training mitigation
strategies and tighter evaluation frameworks to ensure that automated news
content upholds the standards of balanced reporting.

### Cryptography and Security

### 1. [Memory Under Siege: A Comprehensive Survey of Side-Channel Attacks on Memory](http://arxiv.org/pdf/2505.04896v1)

Authors: MD Mahady Hassan, Shanto Roy, Reza Rahaeimehr

Side-channel attacks on memory (SCAM) exploit unintended data leaks from
memory subsystems to infer sensitive information, posing significant threats to
system security. These attacks exploit vulnerabilities in memory access
patterns, cache behaviors, and other microarchitectural features to bypass
traditional security measures. The purpose of this research is to examine SCAM,
classify various attack techniques, and evaluate existing defense mechanisms.
It guides researchers and industry professionals in improving memory security
and mitigating emerging threats. We begin by identifying the major
vulnerabilities in the memory system that are frequently exploited in SCAM,
such as cache timing, speculative execution, \textit{Rowhammer}, and other
sophisticated approaches. Next, we outline a comprehensive taxonomy that
systematically classifies these attacks based on their types, target systems,
attack vectors, and adversarial capabilities required to execute them. In
addition, we review the current landscape of mitigation strategies, emphasizing
their strengths and limitations. This work aims to provide a comprehensive
overview of memory-based side-channel attacks with the goal of providing
significant insights for researchers and practitioners to better understand,
detect, and mitigate SCAM risks.

### 2. [Enhancing Blockchain Cross Chain Interoperability: A Comprehensive Survey](http://arxiv.org/pdf/2505.04934v1)

Authors: Zhihong Deng, Chunming Tang, Taotao Li, Parhat Abla, Qi Chen, Wei Liang, Debiao He

Blockchain technology, introduced in 2008, has revolutionized data storage
and transfer across sectors such as finance, healthcare, intelligent
transportation, and the metaverse. However, the proliferation of blockchain
systems has led to discrepancies in architectures, consensus mechanisms, and
data standards, creating data and value silos that hinder the development of an
integrated multi chain ecosystem. Blockchain interoperability (a.k.a cross
chain interoperability) has thus emerged as a solution to enable seamless data
and asset exchange across disparate blockchains. In this survey, we
systematically analyze over 150 high impact sources from academic journals,
digital libraries, and grey literature to provide an in depth examination of
blockchain interoperability. By exploring the existing methods, technologies,
and architectures, we offer a classification of interoperability approaches
including Atomic Swaps, Sidechains, Light Clients, and so on, which represent
the most comprehensive overview to date. Furthermore, we investigate the
convergence of academic research with industry practices, underscoring the
importance of collaborative efforts in advancing blockchain innovation.
Finally, we identify key strategic insights, challenges, and future research
trajectories in this field. Our findings aim to support researchers,
policymakers, and industry leaders in understanding and harnessing the
transformative potential of blockchain interoperability to address current
challenges and drive forward a cohesive multi-chain ecosystem.

### 3. [SoK: A Taxonomy for Distributed-Ledger-Based Identity Management](http://arxiv.org/pdf/2505.05100v1)

Authors: Awid Vaziry, Sandro Rodriguez Garzon, Patrick Herbke, Carlo Segat, Axel Kupper

The intersection of blockchain (distributed ledger) and identity management
lacks a comprehensive framework for classifying distributed-ledger-based
identity solutions. This paper introduces a methodologically developed taxonomy
derived from the analysis of 390 scientific papers and expert discussions.
  The resulting framework consists of 22 dimensions with 113 characteristics,
organized into three groups: trust anchor implementations, identity
architectures (identifiers and credentials), and ledger specifications. This
taxonomy facilitates the systematic analysis, comparison, and design of
distributed-ledger-based identity solutions, as demonstrated through its
application to two distinct architectures.
  As the first methodology-driven taxonomy in this field, this work advances
standardization and enhances understanding of distributed-ledger-based identity
architectures. It provides researchers and practitioners with a structured
framework for evaluating design decisions and implementation approaches.

### 4. [QUIC-Exfil: Exploiting QUIC's Server Preferred Address Feature to Perform Data Exfiltration Attacks](http://arxiv.org/pdf/2505.05292v1)

Authors: Thomas Grübl, Weijie Niu, Jan von der Assen, Burkhard Stiller

The QUIC protocol is now widely adopted by major tech companies and accounts
for a significant fraction of today's Internet traffic. QUIC's multiplexing
capabilities, encrypted headers, dynamic IP address changes, and encrypted
parameter negotiations make the protocol not only more efficient, secure, and
censorship-resistant, but also practically unmanageable by firewalls. This
opens doors for attackers who may exploit certain traits of the QUIC protocol
to perform targeted attacks, such as data exfiltration attacks. Whereas
existing data exfiltration techniques, such as TLS and DNS-based exfiltration,
can be detected on a firewall level, QUIC-based data exfiltration is more
difficult to detect, since changes in IP addresses and ports are inherent to
the protocol's normal behavior. To show the feasibility of a QUIC-based data
exfiltration attack, we introduce a novel method leveraging the server
preferred address feature of the QUIC protocol and, thus, allows an attacker to
exfiltrate sensitive data from an infected machine to a malicious server,
disguised as a server-side connection migration. The attack is implemented as a
proof of concept tool in Rust. We evaluated the performance of five anomaly
detection classifiers - Random Forest, Multi-Layer Perceptron, Support Vector
Machine, Autoencoder, and Isolation Forest - trained on datasets collected from
three network traffic scenarios. The classifiers were trained on over 700K
benign and malicious QUIC packets and 786 connection migration events, but were
unable to detect the data exfiltration attempts. Furthermore, post-analysis of
the traffic captures did not reveal any identifiable fingerprint. As part of
our evaluation, we also interviewed five leading firewall vendors and found
that, as of today, no major firewall vendor implements functionality capable of
distinguishing between benign and malicious QUIC connection migrations.

### 5. [FedRE: Robust and Effective Federated Learning with Privacy Preference](http://arxiv.org/pdf/2505.04889v1)

Authors: Tianzhe Xiao, Yichen Li, Yu Zhou, Yining Qi, Yi Liu, Wei Wang, Haozhao Wang, Yi Wang, Ruixuan Li

Despite Federated Learning (FL) employing gradient aggregation at the server
for distributed training to prevent the privacy leakage of raw data, private
information can still be divulged through the analysis of uploaded gradients
from clients. Substantial efforts have been made to integrate local
differential privacy (LDP) into the system to achieve a strict privacy
guarantee. However, existing methods fail to take practical issues into account
by merely perturbing each sample with the same mechanism while each client may
have their own privacy preferences on privacy-sensitive information (PSI),
which is not uniformly distributed across the raw data. In such a case,
excessive privacy protection from private-insensitive information can
additionally introduce unnecessary noise, which may degrade the model
performance. In this work, we study the PSI within data and develop FedRE, that
can simultaneously achieve robustness and effectiveness benefits with LDP
protection. More specifically, we first define PSI with regard to the privacy
preferences of each client. Then, we optimize the LDP by allocating less
privacy budget to gradients with higher PSI in a layer-wise manner, thus
providing a stricter privacy guarantee for PSI. Furthermore, to mitigate the
performance degradation caused by LDP, we design a parameter aggregation
mechanism based on the distribution of the perturbed information. We conducted
experiments with text tamper detection on T-SROIE and DocTamper datasets, and
FedRE achieves competitive performance compared to state-of-the-art methods.

### 6. [ChainMarks: Securing DNN Watermark with Cryptographic Chain](http://arxiv.org/pdf/2505.04977v1)

Authors: Brian Choi, Shu Wang, Isabelle Choi, Kun Sun

With the widespread deployment of deep neural network (DNN) models, dynamic
watermarking techniques are being used to protect the intellectual property of
model owners. However, recent studies have shown that existing watermarking
schemes are vulnerable to watermark removal and ambiguity attacks. Besides, the
vague criteria for determining watermark presence further increase the
likelihood of such attacks. In this paper, we propose a secure DNN watermarking
scheme named ChainMarks, which generates secure and robust watermarks by
introducing a cryptographic chain into the trigger inputs and utilizes a
two-phase Monte Carlo method for determining watermark presence. First,
ChainMarks generates trigger inputs as a watermark dataset by repeatedly
applying a hash function over a secret key, where the target labels associated
with trigger inputs are generated from the digital signature of model owner.
Then, the watermarked model is produced by training a DNN over both the
original and watermark datasets. To verify watermarks, we compare the predicted
labels of trigger inputs with the target labels and determine ownership with a
more accurate decision threshold that considers the classification probability
of specific models. Experimental results show that ChainMarks exhibits higher
levels of robustness and security compared to state-of-the-art watermarking
schemes. With a better marginal utility, ChainMarks provides a higher
probability guarantee of watermark presence in DNN models with the same level
of watermark accuracy.

### 7. [A Weighted Byzantine Fault Tolerance Consensus Driven Trusted Multiple Large Language Models Network](http://arxiv.org/pdf/2505.05103v1)

Authors: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dongcheng Zhao, Dusit Niyato, Hongfang Yu, Schahram Dustdar

Large Language Models (LLMs) have achieved remarkable success across a wide
range of applications. However, individual LLMs often produce inconsistent,
biased, or hallucinated outputs due to limitations in their training corpora
and model architectures. Recently, collaborative frameworks such as the
Multi-LLM Network (MultiLLMN) have been introduced, enabling multiple LLMs to
interact and jointly respond to user queries. Nevertheless, MultiLLMN
architectures raise critical concerns regarding the reliability and security of
the generated content, particularly in open environments where malicious or
compromised LLMs may be present. Moreover, reliance on centralized coordination
undermines system efficiency and introduces single points of failure. In this
paper, we propose a novel Trusted MultiLLMN framework, driven by a Weighted
Byzantine Fault Tolerance (WBFT) blockchain consensus mechanism, to ensure the
reliability, security, and efficiency of multi-LLM collaboration. In WBFT,
voting weights are adaptively assigned to each LLM based on its response
quality and trustworthiness, incentivizing reliable behavior, and reducing the
impact of malicious nodes. Extensive simulations demonstrate that WBFT
significantly improves both consensus security and efficiency compared to
classical and modern consensus mechanisms, particularly under wireless network
conditions. Furthermore, our evaluations reveal that Trusted MultiLLMN
supported by WBFT can deliver higher-quality and more credible responses than
both single LLMs and conventional MultiLLMNs, thereby providing a promising
path toward building robust, decentralized AI collaboration networks.

### 8. [FedTDP: A Privacy-Preserving and Unified Framework for Trajectory Data Preparation via Federated Learning](http://arxiv.org/pdf/2505.05155v1)

Authors: Zhihao Zeng, Ziquan Fang, Wei Shao, Lu Chen, Yunjun Gao

Trajectory data, which capture the movement patterns of people and vehicles
over time and space, are crucial for applications like traffic optimization and
urban planning. However, issues such as noise and incompleteness often
compromise data quality, leading to inaccurate trajectory analyses and limiting
the potential of these applications. While Trajectory Data Preparation (TDP)
can enhance data quality, existing methods suffer from two key limitations: (i)
they do not address data privacy concerns, particularly in federated settings
where trajectory data sharing is prohibited, and (ii) they typically design
task-specific models that lack generalizability across diverse TDP scenarios.
To overcome these challenges, we propose FedTDP, a privacy-preserving and
unified framework that leverages the capabilities of Large Language Models
(LLMs) for TDP in federated environments. Specifically, we: (i) design a
trajectory privacy autoencoder to secure data transmission and protect privacy,
(ii) introduce a trajectory knowledge enhancer to improve model learning of
TDP-related knowledge, enabling the development of TDP-oriented LLMs, and (iii)
propose federated parallel optimization to enhance training efficiency by
reducing data transmission and enabling parallel model training. Experiments on
6 real datasets and 10 mainstream TDP tasks demonstrate that FedTDP
consistently outperforms 13 state-of-the-art baselines.

### 9. [SUUM: Timestamp-based Nakamoto-style Blockchains are Vulnerable](http://arxiv.org/pdf/2505.05328v1)

Authors: Junjie Hu, Na Ruan

We introduce two advanced attack strategies, the Unrestricted Uncle Maker
(UUM) Attack and the Staircase-Unrestricted Uncle Maker (SUUM) Attack, which
fundamentally threaten the security of timestamp-based Nakamoto-style
blockchains by inflicting permanent systemic harm. Unlike prior work that
merely enhances adversarial rewards, these attacks exploit vulnerabilities in
timestamp manipulation and fork selection rules to irreversibly destabilize
blockchain fairness and incentive mechanisms. Specifically, the SUUM attack
enables adversaries to persistently launch attacks at zero cost, eliminating
constraints on block withholding and risk-free conditions, while systematically
maximizing rewards through coordinated timestamp adjustments and strategic
block release.
  Our analysis demonstrates that SUUM adversaries achieve disproportionate
reward advantages over both UUM and the original Riskless Uncle Maker (RUM)
Attack [CCS '23], with all three strategies surpassing honest mining.
Crucially, SUUM's cost-free persistence allows adversaries to indefinitely
drain rewards from honest participants by maintaining minimal difficulty risks
through precise timestamp manipulation. This creates a self-reinforcing cycle:
adversaries amplify their profits while suppressing honest returns, thereby
permanently eroding the protocol's security assumptions. Through rigorous
theoretical modeling and simulations, we validate how SUUM's combination of
timestamp tampering, block withholding, and difficulty risk control enables
unmitigated exploitation of consensus mechanisms. This work underscores the
existential risks posed by timestamp-based Nakamoto-style protocols and
advocates urgent countermeasures to ensure long-term stability.

### 10. [Walrus: An Efficient Decentralized Storage Network](http://arxiv.org/pdf/2505.05370v1)

Authors: George Danezis, Giacomo Giuliari, Eleftherios Kokoris Kogias, Markus Legner, Jean-Pierre Smith, Alberto Sonnino, Karl Wüst

Decentralized storage systems face a fundamental trade-off between
replication overhead, recovery efficiency, and security guarantees. Current
approaches either rely on full replication, incurring substantial storage
costs, or employ trivial erasure coding schemes that struggle with efficient
recovery especially under high storage-node churn. We present Walrus, a novel
decentralized blob storage system that addresses these limitations through
multiple technical innovations. At the core of Walrus is RedStuff, a
two-dimensional erasure coding protocol that achieves high security with only
4.5x replication factor, while enabling self-healing recovery that requires
bandwidth proportional to only the lost data $(O(|blob|/n)$ versus $O(|blob|)$
in traditional systems). Crucially, RedStuff is the first protocol to support
storage challenges in asynchronous networks, preventing adversaries from
exploiting network delays to pass verification without actually storing data.
Walrus also introduces a novel multi-stage epoch change protocol that
efficiently handles storage node churn while maintaining uninterrupted
availability during committee transitions. Our system incorporates
authenticated data structures to defend against malicious clients and ensures
data consistency throughout storage and retrieval processes. Experimental
evaluation demonstrates that Walrus achieves practical performance at scale,
making it suitable for a wide range of decentralized applications requiring
high-integrity, available blob storage with reasonable overhead.

### Computer Vision and Pattern Recognition

### 1. [Mix-QSAM: Mixed-Precision Quantization of the Segment Anything Model](http://arxiv.org/pdf/2505.04861v1)

Authors: Navin Ranjan, Andreas Savakis

The Segment Anything Model (SAM) is a popular vision foundation model;
however, its high computational and memory demands make deployment on
resource-constrained devices challenging. While Post-Training Quantization
(PTQ) is a practical approach for reducing computational overhead, existing PTQ
methods rely on fixed bit-width quantization, leading to suboptimal accuracy
and efficiency. To address this limitation, we propose Mix-QSAM, a
mixed-precision PTQ framework for SAM. First, we introduce a layer-wise
importance score, derived using Kullback-Leibler (KL) divergence, to quantify
each layer's contribution to the model's output. Second, we introduce
cross-layer synergy, a novel metric based on causal mutual information, to
capture dependencies between adjacent layers. This ensures that highly
interdependent layers maintain similar bit-widths, preventing abrupt precision
mismatches that degrade feature propagation and numerical stability. Using
these metrics, we formulate an Integer Quadratic Programming (IQP) problem to
determine optimal bit-width allocation under model size and bit-operation
constraints, assigning higher precision to critical layers while minimizing
bit-width in less influential layers. Experimental results demonstrate that
Mix-QSAM consistently outperforms existing PTQ methods on instance segmentation
and object detection tasks, achieving up to 20% higher average precision under
6-bit and 4-bit mixed-precision settings, while maintaining computational
efficiency.

### 2. [OWT: A Foundational Organ-Wise Tokenization Framework for Medical Imaging](http://arxiv.org/pdf/2505.04899v1)

Authors: Sifan Song, Siyeop Yoon, Pengfei Jin, Sekeun Kim, Matthew Tivnan, Yujin Oh, Runqi Meng, Ling Chen, Zhiliang Lyu, Dufan Wu, Ning Guo, Xiang Li, Quanzheng Li

Recent advances in representation learning often rely on holistic, black-box
embeddings that entangle multiple semantic components, limiting
interpretability and generalization. These issues are especially critical in
medical imaging. To address these limitations, we propose an Organ-Wise
Tokenization (OWT) framework with a Token Group-based Reconstruction (TGR)
training paradigm. Unlike conventional approaches that produce holistic
features, OWT explicitly disentangles an image into separable token groups,
each corresponding to a distinct organ or semantic entity. Our design ensures
each token group encapsulates organ-specific information, boosting
interpretability, generalization, and efficiency while allowing fine-grained
control in downstream tasks. Experiments on CT and MRI datasets demonstrate the
effectiveness of OWT in not only achieving strong image reconstruction and
segmentation performance, but also enabling novel semantic-level generation and
retrieval applications that are out of reach for standard holistic embedding
methods. These findings underscore the potential of OWT as a foundational
framework for semantically disentangled representation learning, offering broad
scalability and applicability to real-world medical imaging scenarios and
beyond.

### 3. [Pro2SAM: Mask Prompt to SAM with Grid Points for Weakly Supervised Object Localization](http://arxiv.org/pdf/2505.04905v1)

Authors: Xi Yang, Songsong Duan, Nannan Wang, Xinbo Gao

Weakly Supervised Object Localization (WSOL), which aims to localize objects
by only using image-level labels, has attracted much attention because of its
low annotation cost in real applications. Current studies focus on the Class
Activation Map (CAM) of CNN and the self-attention map of transformer to
identify the region of objects. However, both CAM and self-attention maps can
not learn pixel-level fine-grained information on the foreground objects, which
hinders the further advance of WSOL. To address this problem, we initiatively
leverage the capability of zero-shot generalization and fine-grained
segmentation in Segment Anything Model (SAM) to boost the activation of
integral object regions. Further, to alleviate the semantic ambiguity issue
accrued in single point prompt-based SAM, we propose an innovative mask prompt
to SAM (Pro2SAM) network with grid points for WSOL task. First, we devise a
Global Token Transformer (GTFormer) to generate a coarse-grained foreground map
as a flexible mask prompt, where the GTFormer jointly embeds patch tokens and
novel global tokens to learn foreground semantics. Secondly, we deliver grid
points as dense prompts into SAM to maximize the probability of foreground
mask, which avoids the lack of objects caused by a single point/box prompt.
Finally, we propose a pixel-level similarity metric to come true the mask
matching from mask prompt to SAM, where the mask with the highest score is
viewed as the final localization map. Experiments show that the proposed
Pro2SAM achieves state-of-the-art performance on both CUB-200-2011 and ILSVRC,
with 84.03\% and 66.85\% Top-1 Loc, respectively.

### 4. [GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](http://arxiv.org/pdf/2505.04915v1)

Authors: Tong Wang, Ting Liu, Xiaochao Qu, Chengjing Wu, Luoqi Liu, Xiaolin Hu

Scene text editing, a subfield of image editing, requires modifying texts in
images while preserving style consistency and visual coherence with the
surrounding environment. While diffusion-based methods have shown promise in
text generation, they still struggle to produce high-quality results. These
methods often generate distorted or unrecognizable characters, particularly
when dealing with complex characters like Chinese. In such systems, characters
are composed of intricate stroke patterns and spatial relationships that must
be precisely maintained. We present GlyphMastero, a specialized glyph encoder
designed to guide the latent diffusion model for generating texts with
stroke-level precision. Our key insight is that existing methods, despite using
pretrained OCR models for feature extraction, fail to capture the hierarchical
nature of text structures - from individual strokes to stroke-level
interactions to overall character-level structure. To address this, our glyph
encoder explicitly models and captures the cross-level interactions between
local-level individual characters and global-level text lines through our novel
glyph attention module. Meanwhile, our model implements a feature pyramid
network to fuse the multi-scale OCR backbone features at the global-level.
Through these cross-level and multi-scale fusions, we obtain more detailed
glyph-aware guidance, enabling precise control over the scene text generation
process. Our method achieves an 18.02\% improvement in sentence accuracy over
the state-of-the-art multi-lingual scene text editing baseline, while
simultaneously reducing the text-region Fr\'echet inception distance by
53.28\%.

### 5. [A Simple Detector with Frame Dynamics is a Strong Tracker](http://arxiv.org/pdf/2505.04917v1)

Authors: Chenxu Peng, Chenxu Wang, Minrui Zou, Danyang Li, Zhengpeng Yang, Yimian Dai, Ming-Ming Cheng, Xiang Li

Infrared object tracking plays a crucial role in Anti-Unmanned Aerial Vehicle
(Anti-UAV) applications. Existing trackers often depend on cropped template
regions and have limited motion modeling capabilities, which pose challenges
when dealing with tiny targets. To address this, we propose a simple yet
effective infrared tiny-object tracker that enhances tracking performance by
integrating global detection and motion-aware learning with temporal priors.
Our method is based on object detection and achieves significant improvements
through two key innovations. First, we introduce frame dynamics, leveraging
frame difference and optical flow to encode both prior target features and
motion characteristics at the input level, enabling the model to better
distinguish the target from background clutter. Second, we propose a trajectory
constraint filtering strategy in the post-processing stage, utilizing
spatio-temporal priors to suppress false positives and enhance tracking
robustness. Extensive experiments show that our method consistently outperforms
existing approaches across multiple metrics in challenging infrared UAV
tracking scenarios. Notably, we achieve state-of-the-art performance in the 4th
Anti-UAV Challenge, securing 1st place in Track 1 and 2nd place in Track 2.

### 6. [Canny2Palm: Realistic and Controllable Palmprint Generation for Large-scale Pre-training](http://arxiv.org/pdf/2505.04922v1)

Authors: Xingzeng Lan, Xing Duan, Chen Chen, Weiyu Lin, Bo Wang

Palmprint recognition is a secure and privacy-friendly method of biometric
identification. One of the major challenges to improve palmprint recognition
accuracy is the scarcity of palmprint data. Recently, a popular line of
research revolves around the synthesis of virtual palmprints for large-scale
pre-training purposes. In this paper, we propose a novel synthesis method named
Canny2Palm that extracts palm textures with Canny edge detector and uses them
to condition a Pix2Pix network for realistic palmprint generation. By
re-assembling palmprint textures from different identities, we are able to
create new identities by seeding the generator with new assemblies. Canny2Palm
not only synthesizes realistic data following the distribution of real
palmprints but also enables controllable diversity to generate large-scale new
identities. On open-set palmprint recognition benchmarks, models pre-trained
with Canny2Palm synthetic data outperform the state-of-the-art with up to 7.2%
higher identification accuracy. Moreover, the performance of models pre-trained
with Canny2Palm continues to improve given 10,000 synthetic IDs while those
with existing methods already saturate, demonstrating the potential of our
method for large-scale pre-training.

### 7. [Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping](http://arxiv.org/pdf/2505.04941v1)

Authors: Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He

Accurate building damage assessment using bi-temporal multi-modal remote
sensing images is essential for effective disaster response and recovery
planning. This study proposes a novel Building-Guided Pseudo-Label Learning
Framework to address the challenges of mapping building damage from
pre-disaster optical and post-disaster SAR images. First, we train a series of
building extraction models using pre-disaster optical images and building
labels. To enhance building segmentation, we employ multi-model fusion and
test-time augmentation strategies to generate pseudo-probabilities, followed by
a low-uncertainty pseudo-label training method for further refinement. Next, a
change detection model is trained on bi-temporal cross-modal images and damaged
building labels. To improve damage classification accuracy, we introduce a
building-guided low-uncertainty pseudo-label refinement strategy, which
leverages building priors from the previous step to guide pseudo-label
generation for damaged buildings, reducing uncertainty and enhancing
reliability. Experimental results on the 2025 IEEE GRSS Data Fusion Contest
dataset demonstrate the effectiveness of our approach, which achieved the
highest mIoU score (54.28%) and secured first place in the competition.

### 8. [ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](http://arxiv.org/pdf/2505.04963v1)

Authors: Onkar Susladkar, Gayatri Deshmukh, Yalcin Tur, Ulas Bagci

Synthesizing medical images remains challenging due to limited annotated
pathological data, modality domain gaps, and the complexity of representing
diffuse pathologies such as liver cirrhosis. Existing methods often struggle to
maintain anatomical fidelity while accurately modeling pathological features,
frequently relying on priors derived from natural images or inefficient
multi-step sampling. In this work, we introduce ViCTr (Vital Consistency
Transfer), a novel two-stage framework that combines a rectified flow
trajectory with a Tweedie-corrected diffusion process to achieve high-fidelity,
pathology-aware image synthesis. First, we pretrain ViCTr on the ATLAS-8k
dataset using Elastic Weight Consolidation (EWC) to preserve critical
anatomical structures. We then fine-tune the model adversarially with Low-Rank
Adaptation (LoRA) modules for precise control over pathology severity. By
reformulating Tweedie's formula within a linear trajectory framework, ViCTr
supports one-step sampling, reducing inference from 50 steps to just 4, without
sacrificing anatomical realism. We evaluate ViCTr on BTCV (CT), AMOS (MRI), and
CirrMRI600+ (cirrhosis) datasets. Results demonstrate state-of-the-art
performance, achieving a Medical Frechet Inception Distance (MFID) of 17.01 for
cirrhosis synthesis 28% lower than existing approaches and improving nnUNet
segmentation by +3.8% mDSC when used for data augmentation. Radiologist reviews
indicate that ViCTr-generated liver cirrhosis MRIs are clinically
indistinguishable from real scans. To our knowledge, ViCTr is the first method
to provide fine-grained, pathology-aware MRI synthesis with graded severity
control, closing a critical gap in AI-driven medical imaging research.

### 9. [CAG-VLM: Fine-Tuning of a Large-Scale Model to Recognize Angiographic Images for Next-Generation Diagnostic Systems](http://arxiv.org/pdf/2505.04964v1)

Authors: Yuto Nakamura, Satoshi Kodera, Haruki Settai, Hiroki Shinohara, Masatsugu Tamura, Tomohiro Noguchi, Tatsuki Furusawa, Ryo Takizawa, Tempei Kabayama, Norihiko Takeda

Coronary angiography (CAG) is the gold-standard imaging modality for
evaluating coronary artery disease, but its interpretation and subsequent
treatment planning rely heavily on expert cardiologists. To enable AI-based
decision support, we introduce a two-stage, physician-curated pipeline and a
bilingual (Japanese/English) CAG image-report dataset. First, we sample 14,686
frames from 539 exams and annotate them for key-frame detection and left/right
laterality; a ConvNeXt-Base CNN trained on this data achieves 0.96 F1 on
laterality classification, even on low-contrast frames. Second, we apply the
CNN to 243 independent exams, extract 1,114 key frames, and pair each with its
pre-procedure report and expert-validated diagnostic and treatment summary,
yielding a parallel corpus. We then fine-tune three open-source VLMs
(PaliGemma2, Gemma3, and ConceptCLIP-enhanced Gemma3) via LoRA and evaluate
them using VLScore and cardiologist review. Although PaliGemma2 w/LoRA attains
the highest VLScore, Gemma3 w/LoRA achieves the top clinician rating (mean
7.20/10); we designate this best-performing model as CAG-VLM. These results
demonstrate that specialized, fine-tuned VLMs can effectively assist
cardiologists in generating clinical reports and treatment recommendations from
CAG images.

### 10. [DenseGrounding: Improving Dense Language-Vision Semantics for Ego-Centric 3D Visual Grounding](http://arxiv.org/pdf/2505.04965v1)

Authors: Henry Zheng, Hao Shi, Qihang Peng, Yong Xien Chng, Rui Huang, Yepeng Weng, Zhongchao Shi, Gao Huang

Enabling intelligent agents to comprehend and interact with 3D environments
through natural language is crucial for advancing robotics and human-computer
interaction. A fundamental task in this field is ego-centric 3D visual
grounding, where agents locate target objects in real-world 3D spaces based on
verbal descriptions. However, this task faces two significant challenges: (1)
loss of fine-grained visual semantics due to sparse fusion of point clouds with
ego-centric multi-view images, (2) limited textual semantic context due to
arbitrary language descriptions. We propose DenseGrounding, a novel approach
designed to address these issues by enhancing both visual and textual
semantics. For visual features, we introduce the Hierarchical Scene Semantic
Enhancer, which retains dense semantics by capturing fine-grained global scene
features and facilitating cross-modal alignment. For text descriptions, we
propose a Language Semantic Enhancer that leverages large language models to
provide rich context and diverse language descriptions with additional context
during model training. Extensive experiments show that DenseGrounding
significantly outperforms existing methods in overall accuracy, with
improvements of 5.81% and 7.56% when trained on the comprehensive full dataset
and smaller mini subset, respectively, further advancing the SOTA in egocentric
3D visual grounding. Our method also achieves 1st place and receives the
Innovation Award in the CVPR 2024 Autonomous Grand Challenge Multi-view 3D
Visual Grounding Track, validating its effectiveness and robustness.

### Computers and Society

### 1. [Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards](http://arxiv.org/pdf/2505.04966v1)

Authors: Jaeho Kim, Yunseok Lee, Seulki Lee

The peer review process in major artificial intelligence (AI) conferences
faces unprecedented challenges with the surge of paper submissions (exceeding
10,000 submissions per venue), accompanied by growing concerns over review
quality and reviewer responsibility. This position paper argues for the need to
transform the traditional one-way review system into a bi-directional feedback
loop where authors evaluate review quality and reviewers earn formal
accreditation, creating an accountability framework that promotes a
sustainable, high-quality peer review system. The current review system can be
viewed as an interaction between three parties: the authors, reviewers, and
system (i.e., conference), where we posit that all three parties share
responsibility for the current problems. However, issues with authors can only
be addressed through policy enforcement and detection tools, and ethical
concerns can only be corrected through self-reflection. As such, this paper
focuses on reforming reviewer accountability with systematic rewards through
two key mechanisms: (1) a two-stage bi-directional review system that allows
authors to evaluate reviews while minimizing retaliatory behavior, (2)a
systematic reviewer reward system that incentivizes quality reviewing. We ask
for the community's strong interest in these problems and the reforms that are
needed to enhance the peer review process.

### 2. [Societal and technological progress as sewing an ever-growing, ever-changing, patchy, and polychrome quilt](http://arxiv.org/pdf/2505.05197v1)

Authors: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Sébastien Krier, Manfred Diaz, Simon Osindero

Artificial Intelligence (AI) systems are increasingly placed in positions
where their decisions have real consequences, e.g., moderating online spaces,
conducting research, and advising on policy. Ensuring they operate in a safe
and ethically acceptable fashion is thus critical. However, most solutions have
been a form of one-size-fits-all "alignment". We are worried that such systems,
which overlook enduring moral diversity, will spark resistance, erode trust,
and destabilize our institutions. This paper traces the underlying problem to
an often-unstated Axiom of Rational Convergence: the idea that under ideal
conditions, rational agents will converge in the limit of conversation on a
single ethics. Treating that premise as both optional and doubtful, we propose
what we call the appropriateness framework: an alternative approach grounded in
conflict theory, cultural evolution, multi-agent systems, and institutional
economics. The appropriateness framework treats persistent disagreement as the
normal case and designs for it by applying four principles: (1) contextual
grounding, (2) community customization, (3) continual adaptation, and (4)
polycentric governance. We argue here that adopting these design principles is
a good way to shift the main alignment metaphor from moral unification to a
more productive metaphor of conflict management, and that taking this step is
both desirable and urgent.

### 3. [Facets of Disparate Impact: Evaluating Legally Consistent Bias in Machine Learning](http://arxiv.org/pdf/2505.05471v1)

Authors: Jarren Briscoe, Assefaw Gebremedhin

Leveraging current legal standards, we define bias through the lens of
marginal benefits and objective testing with the novel metric "Objective
Fairness Index". This index combines the contextual nuances of objective
testing with metric stability, providing a legally consistent and reliable
measure. Utilizing the Objective Fairness Index, we provide fresh insights into
sensitive machine learning applications, such as COMPAS (recidivism
prediction), highlighting the metric's practical and theoretical significance.
The Objective Fairness Index allows one to differentiate between discriminatory
tests and systemic disparities.

### 4. [Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects](http://arxiv.org/pdf/2505.05318v1)

Authors: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci

The rapid adoption of Vision Language Models (VLMs), pre-trained on large
image-text and video-text datasets, calls for protecting and informing users
about when to trust these systems. This survey reviews studies on trust
dynamics in user-VLM interactions, through a multi-disciplinary taxonomy
encompassing different cognitive science capabilities, collaboration modes, and
agent behaviours. Literature insights and findings from a workshop with
prospective VLM users inform preliminary requirements for future VLM trust
studies.

### Databases

### 1. [Spatially Disaggregated Energy Consumption and Emissions in End-use Sectors for Germany and Spain](http://arxiv.org/pdf/2505.05139v1)

Authors: Shruthi Patil, Noah Pflugradt, Jann M. Weinand, Jürgen Kropp, Detlef Stolten

High-resolution energy consumption and emissions datasets are essential for
localized policy-making, resource optimization, and climate action planning.
They enable municipalities to monitor mitigation strategies and foster
engagement among governments, businesses, and communities. However, smaller
municipalities often face data limitations that hinder tailored climate
strategies. This study generates detailed final energy consumption and
emissions data at the local administrative level for Germany and Spain. Using
national datasets, we apply spatial disaggregation techniques with open data
sources. A key innovation is the application of XGBoost for imputing missing
data, combined with a stepwise spatial disaggregation process incorporating
district- and province-level statistics. Prioritizing reproducibility, our
open-data approach provides a scalable framework for municipalities to develop
actionable climate plans. To ensure transparency, we assess the reliability of
imputed values and assign confidence ratings to the disaggregated data.

### 2. [HEXGEN-TEXT2SQL: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL Workflow](http://arxiv.org/pdf/2505.05286v1)

Authors: You Peng, Youhe Jiang, Chen Wang, Binhang Yuan

Recent advances in leveraging the agentic paradigm of large language models
(LLMs) utilization have significantly enhanced Text-to-SQL capabilities,
enabling users without specialized database expertise to query data
intuitively. However, deploying these agentic LLM-based Text-to-SQL systems in
production poses substantial challenges due to their inherently multi-stage
workflows, stringent latency constraints, and potentially heterogeneous GPU
infrastructure in enterprise environments. Current LLM serving frameworks lack
effective mechanisms for handling interdependent inference tasks, dynamic
latency variability, and resource heterogeneity, leading to suboptimal
performance and frequent service-level objective (SLO) violations. In this
paper, we introduce HEXGEN-TEXT2SQL, a novel framework designed explicitly to
schedule and execute agentic multi-stage LLM-based Text-to-SQL workflows on
heterogeneous GPU clusters that handle multi-tenant end-to-end queries.
HEXGEN-TEXT2SQL introduce a hierarchical scheduling approach combining global
workload-balanced task dispatching and local adaptive urgency-guided
prioritization, guided by a systematic analysis of agentic Text-to-SQL
workflows. Additionally, we propose a lightweight simulation-based method for
tuning critical scheduling hyperparameters, further enhancing robustness and
adaptability. Our extensive evaluation on realistic Text-to-SQL benchmarks
demonstrates that HEXGEN-TEXT2SQL significantly outperforms state-of-the-art
LLM serving frameworks. Specifically, HEXGEN-TEXT2SQL reduces latency deadlines
by up to 1.67$\times$ (average: 1.41$\times$) and improves system throughput by
up to 1.75$\times$ (average: 1.65$\times$) compared to vLLM under diverse,
realistic workload conditions. Our code is available at
https://github.com/Relaxed-System-Lab/Hexgen-Flow.

### 3. [Enhancing Text2Cypher with Schema Filtering](http://arxiv.org/pdf/2505.05118v1)

Authors: Makbule Gulcin Ozsoy

Knowledge graphs represent complex data using nodes, relationships, and
properties. Cypher, a powerful query language for graph databases, enables
efficient modeling and querying. Recent advancements in large language models
allow translation of natural language questions into Cypher queries -
Text2Cypher. A common approach is incorporating database schema into prompts.
However, complex schemas can introduce noise, increase hallucinations, and
raise computational costs. Schema filtering addresses these challenges by
including only relevant schema elements, improving query generation while
reducing token costs. This work explores various schema filtering methods for
Text2Cypher task and analyzes their impact on token length, performance, and
cost. Results show that schema filtering effectively optimizes Text2Cypher,
especially for smaller models. Consistent with prior research, we find that
larger models benefit less from schema filtering due to their longer context
capabilities. However, schema filtering remains valuable for both larger and
smaller models in cost reduction.

### 4. [Text2Cypher: Data Pruning using Hard Example Selection](http://arxiv.org/pdf/2505.05122v1)

Authors: Makbule Gulcin Ozsoy

Database query languages such as SQL for relational databases and Cypher for
graph databases have been widely adopted. Recent advancements in large language
models (LLMs) enable natural language interactions with databases through
models like Text2SQL and Text2Cypher. Fine-tuning these models typically
requires large, diverse datasets containing non-trivial examples. However, as
dataset size increases, the cost of fine-tuning also rises. This makes smaller,
high-quality datasets essential for reducing costs for the same or better
performance. In this paper, we propose five hard-example selection techniques
for pruning the Text2Cypher dataset, aiming to preserve or improve performance
while reducing resource usage. Our results show that these hard-example
selection approaches can halve training time and costs with minimal impact on
performance, and demonstrates that hard-example selection provides a
cost-effective solution.

### Distributed, Parallel, and Cluster Computing

### 1. [DFPL: Decentralized Federated Prototype Learning Across Heterogeneous Data Distributions](http://arxiv.org/pdf/2505.04947v1)

Authors: Hongliang Zhang, Fenghua Xu, Zhongyuan Yu, Chunqiang Hu, Shanchen Pang, Xiaofen Wang, Jiguo Yu

Federated learning is a distributed machine learning paradigm that enables
the collaborative training of multiple clients through centralized model
aggregation. However, standard federated learning relies on a centralized
server, making it vulnerable to server failures. While existing solutions
utilize blockchain technology to implement Decentralized Federated Learning
(DFL), the statistical heterogeneity of data distributions among clients
severely degrades the DFL's performance. Driven by this issue, this paper
proposes a decentralized federated prototype learning framework, named DFPL,
which significantly improves the performance of distributed machine learning
across heterogeneous data distributions. Specifically, our framework introduces
prototype learning into DFL to address statistical heterogeneity, which greatly
reduces the number of parameters exchanged between clients. Additionally,
blockchain is embedded into our framework, enabling the training and mining
processes to be implemented at each client. From a theoretical perspective, we
provide convergence guarantee of DFPL by combining resource allocation for
training and mining. The experiments highlight the superiority of our DFPL
framework in communication efficiency and test performance across three
benchmark datasets with heterogeneous data distributions.

### 2. [CacheFL: Efficient Federated Cache Model Fine-Tuning for Vision-Language Models](http://arxiv.org/pdf/2505.05130v1)

Authors: Mengjun Yi, Hanwen Zhang, Hui Dou, Jian Zhao, Furao Shen

Large pre-trained Vision-Language Models (VLMs), such as Contrastive
Language-Image Pre-training (CLIP), have exhibited remarkable zero-shot
performance across various image classification tasks. Fine-tuning these models
on domain-specific datasets further enhances their effectiveness for downstream
applications. However, fine-tuning in cloud environments raises significant
concerns regarding data security and privacy. Federated Learning (FL) offers a
decentralized solution by enabling model training across local clients without
centralizing sensitive data, but the high communication and computation costs
of transmitting full pre-trained models during training limit its scalability.
Additionally, non-Independent and Identically Distributed (non-IID) data across
local clients can negatively impact model convergence and performance. To
address these challenges, we propose CacheFL, a novel federated learning method
that replaces traditional full model fine-tuning with lightweight cache model
fine-tuning. The cache model is initialized using a class-balanced dataset
generated by a generative pre-trained model, effectively mitigating the impact
of non-IID data. This cache model is then distributed to local clients for
fine-tuning, and the updated parameters from each client are aggregated on the
server and redistributed. With the updated cache model, the classification
performance of CLIP is improved after just a few epochs. By limiting the
training and communication to the cache model, CacheFL significantly reduces
resource demands while ensuring data privacy and security. Extensive
experiments conducted on ImageNet and 10 additional datasets demonstrate that
CacheFL outperforms traditional approaches in terms of classification accuracy,
resource efficiency, and privacy preservation.

### 3. [PUDTune: Multi-Level Charging for High-Precision Calibration in Processing-Using-DRAM](http://arxiv.org/pdf/2505.05266v1)

Authors: Tatsuya Kubo, Daichi Tokuda, Lei Qu, Ting Cao, Shinya Takamaeda-Yamazaki

Recently, practical analog in-memory computing has been realized using
unmodified commercial DRAM modules. The underlying Processing-Using-DRAM (PUD)
techniques enable high-throughput bitwise operations directly within DRAM
arrays. However, the presence of inherent error-prone columns hinders PUD's
practical adoption. While selectively using only error-free columns would
ensure reliability, this approach significantly reduces PUD's computational
throughput.
  This paper presents PUDTune, a novel high-precision calibration technique for
increasing the number of error-free columns in PUD. PUDTune compensates for
errors by applying pre-identified column-specific offsets to PUD operations. By
leveraging multi-level charge states of DRAM cells, PUDTune generates
fine-grained and wide-range offset variations despite the limited available
rows. Our experiments with DDR4 DRAM demonstrate that PUDTune increases the
number of error-free columns by 1.81$\times$ compared to conventional
implementations, improving addition and multiplication throughput by
1.88$\times$ and 1.89$\times$ respectively.

### 4. [Empirical Analysis of Transaction Conflicts in Ethereum and Solana for Parallel Execution](http://arxiv.org/pdf/2505.05358v1)

Authors: Parwat Singh Anjana, Srivatsan Ravi

This paper presents a comprehensive analysis of historical data across two
popular blockchain networks: Ethereum and Solana. Our study focuses on two key
aspects: transaction conflicts and the maximum theoretical parallelism within
historical blocks. We aim to quantify the degree of transaction parallelism and
assess how effectively it can be exploited by systematically examining
block-level characteristics, both within individual blocks and across different
historical periods. In particular, this study is the first of its kind to
leverage historical transactional workloads to evaluate transactional conflict
patterns. By offering a structured approach to analyzing these conflicts, our
research provides valuable insights and an empirical basis for developing more
efficient parallel execution techniques in the Ethereum and Solana Virtual
Machines. Our empirical analysis reveals that Ethereum blocks frequently
achieve high independence$-$over 50\% in more than 50\% of blocks, while Solana
blocks contain longer conflict chains, comprising $\sim$59\% of the block size
compared to $\sim$18\% in Ethereum, reflecting fundamentally different parallel
execution dynamics.

### 5. [Walrus: An Efficient Decentralized Storage Network](http://arxiv.org/pdf/2505.05370v1)

Authors: George Danezis, Giacomo Giuliari, Eleftherios Kokoris Kogias, Markus Legner, Jean-Pierre Smith, Alberto Sonnino, Karl Wüst

Decentralized storage systems face a fundamental trade-off between
replication overhead, recovery efficiency, and security guarantees. Current
approaches either rely on full replication, incurring substantial storage
costs, or employ trivial erasure coding schemes that struggle with efficient
recovery especially under high storage-node churn. We present Walrus, a novel
decentralized blob storage system that addresses these limitations through
multiple technical innovations. At the core of Walrus is RedStuff, a
two-dimensional erasure coding protocol that achieves high security with only
4.5x replication factor, while enabling self-healing recovery that requires
bandwidth proportional to only the lost data $(O(|blob|/n)$ versus $O(|blob|)$
in traditional systems). Crucially, RedStuff is the first protocol to support
storage challenges in asynchronous networks, preventing adversaries from
exploiting network delays to pass verification without actually storing data.
Walrus also introduces a novel multi-stage epoch change protocol that
efficiently handles storage node churn while maintaining uninterrupted
availability during committee transitions. Our system incorporates
authenticated data structures to defend against malicious clients and ensures
data consistency throughout storage and retrieval processes. Experimental
evaluation demonstrates that Walrus achieves practical performance at scale,
making it suitable for a wide range of decentralized applications requiring
high-integrity, available blob storage with reasonable overhead.

### 6. [Empowering Scientific Workflows with Federated Agents](http://arxiv.org/pdf/2505.05428v1)

Authors: J. Gregory Pauloski, Yadu Babuji, Ryan Chard, Mansi Sakarvadia, Kyle Chard, Ian Foster

Agentic systems, in which diverse agents cooperate to tackle challenging
problems, are exploding in popularity in the AI community. However, the agentic
frameworks used to build these systems have not previously enabled use with
research cyberinfrastructure. Here we introduce Academy, a modular and
extensible middleware designed to deploy autonomous agents across the federated
research ecosystem, including HPC systems, experimental facilities, and data
repositories. To meet the demands of scientific computing, Academy supports
asynchronous execution, heterogeneous resources, high-throughput data flows,
and dynamic resource availability. It provides abstractions for expressing
stateful agents, managing inter-agent coordination, and integrating computation
with experimental control. We present microbenchmark results that demonstrate
high performance and scalability in HPC environments. To demonstrate the
breadth of applications that can be supported by agentic workflow designs, we
also present case studies in materials discovery, decentralized learning, and
information extraction in which agents are deployed across diverse HPC systems.

### 7. [Federated Learning for Cyber Physical Systems: A Comprehensive Survey](http://arxiv.org/pdf/2505.04873v1)

Authors: Minh K. Quan, Pubudu N. Pathirana, Mayuri Wijayasundara, Sujeeva Setunge, Dinh C. Nguyen, Christopher G. Brinton, David J. Love, H. Vincent Poor

The integration of machine learning (ML) in cyber physical systems (CPS) is a
complex task due to the challenges that arise in terms of real-time decision
making, safety, reliability, device heterogeneity, and data privacy. There are
also open research questions that must be addressed in order to fully realize
the potential of ML in CPS. Federated learning (FL), a distributed approach to
ML, has become increasingly popular in recent years. It allows models to be
trained using data from decentralized sources. This approach has been gaining
popularity in the CPS field, as it integrates computer, communication, and
physical processes. Therefore, the purpose of this work is to provide a
comprehensive analysis of the most recent developments of FL-CPS, including the
numerous application areas, system topologies, and algorithms developed in
recent years. The paper starts by discussing recent advances in both FL and
CPS, followed by their integration. Then, the paper compares the application of
FL in CPS with its applications in the internet of things (IoT) in further
depth to show their connections and distinctions. Furthermore, the article
scrutinizes how FL is utilized in critical CPS applications, e.g., intelligent
transportation systems, cybersecurity services, smart cities, and smart
healthcare solutions. The study also includes critical insights and lessons
learned from various FL-CPS implementations. The paper's concluding section
delves into significant concerns and suggests avenues for further research in
this fast-paced and dynamic era.

### Discrete Mathematics

### 1. [Sideways on the highways](http://arxiv.org/pdf/2505.05426v1)

Authors: Victor Lutfalla

We present two generalised ants (LLRRRL and LLRLRLL) which admit both highway
behaviours and other kinds of emergent behaviours from initially finite
configurations. This limits the well known Highway conjecture on Langton's ant
as it shows that a generalised version of this conjecture generically does not
hold on generalised ants.

### 2. [p-complete square-free Word-representation of Word-representable Graphs](http://arxiv.org/pdf/2505.05110v1)

Authors: Biswajit Das, Ramesh Hariharasubramanian

A graph $G = (V,E)$ is word-representable, if there exists a word $w$ over
the alphabet $V$ such that for letters ${x,y} \in V$ , $x$ and $y$ alternate in
$w$ if and only if $xy$ is an edge in the graph $G$. In this paper, we
introduce the concept of $p$-complete square-free word-representable graph
$G(V,E)$. A word $w$ defined over alphabet $V$ is called $p$-complete
square-free word if there does not exist any subset $S\subseteq \Sigma$ such
that the word $w_{S}$ contains a square $XX$ where $|X| \ge p$ and $1\le p \le
|w|/2$. A word-representable graph is considered $p$-complete square-free
word-representable if there exists a $p$-complete square-free word-representant
of that graph. This pattern is significant as it proves the existence of
patterns that do not depend on graph labelling and cannot be avoided by certain
classes of word-representable graphs. The class of word-representable graphs
includes both $p$-complete square-free word-representable graphs and
non-$p$-complete square-free word-representable graphs. Additionally, this
concept generalises the square pattern found in the words. A word-representable
graph is $p$-complete square-free uniform word-representable if its
$p$-complete square-free word-representant is a uniform word. We analyse the
properties of $p$-complete square-free uniform words and find that the graphs
represented by these words avoid having $K_p$ (the complete graph on $p$
vertices) as an induced subgraph. We provide classifications for small values
of $p$: for $p=1$, only complete graphs and for $p=2$, only complete and
edgeless graphs satisfy the condition. We find that $K_3$-free circle graphs
are 3-complete square-free uniform word-representable. Furthermore, we
establish that only graphs with representation number at most 3 can be
3-complete square-free uniform word-representable and provide a constructive
method to generate such graphs.

### Data Structures and Algorithms

### 1. [PSSketch: Finding Persistent and Sparse Flow with High Accuracy and Efficiency](http://arxiv.org/pdf/2505.04892v1)

Authors: Jiayao Wang, Qilong Shi, Xiyan Liang, Han Wang, Wenjun Li, Ziling Wei, Weizhe Zhang, Shuhui Chen

Finding persistent sparse (PS) flow is critical to early warning of many
threats. Previous works have predominantly focused on either heavy or
persistent flows, with limited attention given to PS flows. Although some
recent studies pay attention to PS flows, they struggle to establish an
objective criterion due to insufficient data-driven observations, resulting in
reduced accuracy. In this paper, we define a new criterion "anomaly boundary"
to distinguish PS flows from regular flows. Specifically, a flow whose
persistence exceeds a threshold will be protected, while a protected flow with
a density lower than a threshold is reported as a PS flow. We then introduce
PSSketch, a high-precision layered sketch to find PS flows. PSSketch employs
variable-length bitwise counters, where the first layer tracks the frequency
and persistence of all flows, and the second layer protects potential PS flows
and records overflow counts from the first layer. Some optimizations have also
been implemented to reduce memory consumption further and improve accuracy. The
experiments show that PSSketch reduces memory consumption by an order of
magnitude compared to the strawman solution combined with existing work.
Compared with SOTA solutions for finding PS flows, it outperforms up to 2.94x
in F1 score and reduces ARE by 1-2 orders of magnitude. Meanwhile, PSSketch
achieves a higher throughput than these solutions.

### 2. [With a Little Help From My Friends: Exploiting Probability Distribution Advice in Algorithm Design](http://arxiv.org/pdf/2505.04949v1)

Authors: Clément L. Canonne, Kenny Chen, Julián Mestre

We study online algorithms with predictions using distributional advice, a
type of prediction that arises when leveraging expert knowledge or historical
data. To demonstrate the usefulness and versatility of this framework, we focus
on two fundamental problems: first, the prophet inequality problem, for which
we provide an algorithm achieving
$\max\{\frac{1}{2}-\eta-o(1),\frac{1}{e}\}$-competitive ratio, where $\eta$
quantifies the quality of the prediction. Second, we turn to the online metric
matching problem under random arrivals, for which our main positive result is
an algorithm achieving the optimal cost under perfect advice, while smoothly
defaulting to competitive ratios comparable to advice-free algorithms as the
prediction's quality degrades.

### 3. [Zip-Tries: Simple Dynamic Data Structures for Strings](http://arxiv.org/pdf/2505.04953v1)

Authors: David Eppstein, Ofek Gila, Michael T. Goodrich, Ryuto Kitagawa

In this paper, we introduce zip-tries, which are simple, dynamic,
memory-efficient data structures for strings. Zip-tries support search and
update operations for $k$-length strings in $\mathcal{O}(k+\log n)$ time in the
standard RAM model or in $\mathcal{O}(k/\alpha+\log n)$ time in the word RAM
model, where $\alpha$ is the length of the longest string that can fit in a
memory word, and $n$ is the number of strings in the trie. Importantly, we show
how zip-tries can achieve this while only requiring $\mathcal{O}(\log{\log{n}}
+ \log{\log{\frac{k}{\alpha}}})$ bits of metadata per node w.h.p., which is an
exponential improvement over previous results for long strings. Despite being
considerably simpler and more memory efficient, we show how zip-tries perform
competitively with state-of-the-art data structures on large datasets of long
strings.
  Furthermore, we provide a simple, general framework for parallelizing string
comparison operations in linked data structures, which we apply to zip-tries to
obtain parallel zip-tries. Parallel zip-tries are able to achieve good search
and update performance in parallel, performing such operations in
$\mathcal{O}(\log{n})$ span. We also apply our techniques to an existing
external-memory string data structure, the string B-tree, obtaining a parallel
string B-tree which performs search operations using $\mathcal{O}(\log_B{n})$
I/O span and $\mathcal{O}(\frac{k}{\alpha B} + \log_B{n})$ I/O work in the
parallel external memory (PEM) model. The parallel string B-tree can perform
prefix searches using only $\mathcal{O}(\frac{\log{n}}{\log{\log{n}}})$ span
under the practical PRAM model.
  For the case of long strings that share short common prefixes, we provide
LCP-aware variants of all our algorithms that should be quite efficient in
practice, which we justify empirically.

### 4. [Learning Partitions with Optimal Query and Round Complexities](http://arxiv.org/pdf/2505.05009v1)

Authors: Hadley Black, Arya Mazumdar, Barna Saha

We consider the basic problem of learning an unknown partition of $n$
elements into at most $k$ sets using simple queries that reveal information
about a small subset of elements. Our starting point is the well-studied
pairwise same-set queries which ask if a pair of elements belong to the same
class. It is known that non-adaptive algorithms require $\Theta(n^2)$ queries,
while adaptive algorithms require $\Theta(nk)$ queries, and the best known
algorithm uses $k-1$ rounds. This problem has been studied extensively over the
last two decades in multiple communities due to its fundamental nature and
relevance to clustering, active learning, and crowd sourcing. In many
applications, it is of high interest to reduce adaptivity while minimizing
query complexity. We give a complete characterization of the deterministic
query complexity of this problem as a function of the number of rounds, $r$,
interpolating between the non-adaptive and adaptive settings: for any constant
$r$, the query complexity is
$\Theta(n^{1+\frac{1}{2^r-1}}k^{1-\frac{1}{2^r-1}})$. Our algorithm only needs
$O(\log \log n)$ rounds to attain the optimal $O(nk)$ query complexity.
  Next, we consider two generalizations of pairwise queries to subsets $S$ of
size at most $s$: (1) weak subset queries which return the number of classes
intersected by $S$, and (2) strong subset queries which return the entire
partition restricted on $S$. Once again in crowd sourcing applications, queries
on large sets may be prohibitive. For non-adaptive algorithms, we show
$\Omega(n^2/s^2)$ strong queries are needed. Perhaps surprisingly, we show that
there is a non-adaptive algorithm using weak queries that matches this bound up
to log-factors for all $s \leq \sqrt{n}$. More generally, we obtain nearly
matching upper and lower bounds for algorithms using subset queries in terms of
both the number of rounds, $r$, and the query size bound, $s$.

### 5. [Efficient Parallel Ising Samplers via Localization Schemes](http://arxiv.org/pdf/2505.05185v1)

Authors: Xiaoyu Chen, Hongyang Liu, Yitong Yin, Xinyuan Zhang

We introduce efficient parallel algorithms for sampling from the Gibbs
distribution and estimating the partition function of Ising models. These
algorithms achieve parallel efficiency, with polylogarithmic depth and
polynomial total work, and are applicable to Ising models in the following
regimes: (1) Ferromagnetic Ising models with external fields; (2) Ising models
with interaction matrix $J$ of operator norm $\|J\|_2<1$.
  Our parallel Gibbs sampling approaches are based on localization schemes,
which have proven highly effective in establishing rapid mixing of Gibbs
sampling. In this work, we employ two such localization schemes to obtain
efficient parallel Ising samplers: the \emph{field dynamics} induced by
\emph{negative-field localization}, and \emph{restricted Gaussian dynamics}
induced by \emph{stochastic localization}. This shows that localization schemes
are powerful tools, not only for achieving rapid mixing but also for the
efficient parallelization of Gibbs sampling.

### 6. [Overlapping Biclustering](http://arxiv.org/pdf/2505.05213v1)

Authors: Matthias Bentert, Pål Grønås Drange, Erlend Haugen

In this paper, we introduce Bicluster Editing with Vertex Splitting, a
variant of the Bicluster Editing problem, which aims to transform a given graph
into a bicluster graph using a minimum number of operations. In Bicluster
Editing, the allowed operations are the insertion and deletion of edges. In
Bicluster Editing with Vertex Splitting we additionally allow overlapping
clusters, which we model by allowing vertex splits. We prove that Bicluster
Editing with Vertex Splitting is NP-complete. On the positive side, we show
that it admits a polynomial kernel with respect to the number k of allowed edit
operations and present an algorithm running in O(k^{11k} + n + m) time, where n
and m denote the number of vertices and edges in the input graph, respectively.

### 7. [InfTDA: A Simple TopDown Mechanism for Hierarchical Differentially Private Counting Queries](http://arxiv.org/pdf/2505.05347v1)

Authors: Fabrizio Boninsegna

This paper extends $\texttt{InfTDA}$, a mechanism proposed in (Boninsegna,
Silvestri, PETS 2025) for mobility datasets with origin and destination trips,
in a general setting. The algorithm presented in this paper works for any
dataset of $d$ categorical features and produces a differentially private
synthetic dataset that answers all hierarchical queries, a special case of
marginals, each with bounded maximum absolute error. The algorithm builds upon
the TopDown mechanism developed for the 2020 US Census.

### 8. [CART-ELC: Oblique Decision Tree Induction via Exhaustive Search](http://arxiv.org/pdf/2505.05402v1)

Authors: Andrew D. Laack

Oblique decision trees have attracted attention due to their potential for
improved classification performance over traditional axis-aligned decision
trees. However, methods that rely on exhaustive search to find oblique splits
face computational challenges. As a result, they have not been widely explored.
We introduce a novel algorithm, Classification and Regression Tree - Exhaustive
Linear Combinations (CART-ELC), for inducing oblique decision trees that
performs an exhaustive search on a restricted set of hyperplanes. We then
investigate the algorithm's computational complexity and its predictive
capabilities. Our results demonstrate that CART-ELC consistently achieves
competitive performance on small datasets, often yielding statistically
significant improvements in classification accuracy relative to existing
decision tree induction algorithms, while frequently producing shallower,
simpler, and thus more interpretable trees.

### Emerging Technologies

### 1. [Empirical Analysis of Transaction Conflicts in Ethereum and Solana for Parallel Execution](http://arxiv.org/pdf/2505.05358v1)

Authors: Parwat Singh Anjana, Srivatsan Ravi

This paper presents a comprehensive analysis of historical data across two
popular blockchain networks: Ethereum and Solana. Our study focuses on two key
aspects: transaction conflicts and the maximum theoretical parallelism within
historical blocks. We aim to quantify the degree of transaction parallelism and
assess how effectively it can be exploited by systematically examining
block-level characteristics, both within individual blocks and across different
historical periods. In particular, this study is the first of its kind to
leverage historical transactional workloads to evaluate transactional conflict
patterns. By offering a structured approach to analyzing these conflicts, our
research provides valuable insights and an empirical basis for developing more
efficient parallel execution techniques in the Ethereum and Solana Virtual
Machines. Our empirical analysis reveals that Ethereum blocks frequently
achieve high independence$-$over 50\% in more than 50\% of blocks, while Solana
blocks contain longer conflict chains, comprising $\sim$59\% of the block size
compared to $\sim$18\% in Ethereum, reflecting fundamentally different parallel
execution dynamics.

### 2. [Integrating Communication, Sensing, and Security: Progress and Prospects of PLS in ISAC Systems](http://arxiv.org/pdf/2505.05090v1)

Authors: Waqas Aman, El-Mehdi Illi, Marwa Qaraqe, Saif Al-Kuwari

The sixth generation of wireless networks defined several key performance
indicators (KPIs) for assessing its networks, mainly in terms of reliability,
coverage, and sensing. In this regard, remarkable attention has been paid
recently to the integrated sensing and communication (ISAC) paradigm as an
enabler for efficiently and jointly performing communication and sensing using
the same spectrum and hardware resources. On the other hand, ensuring
communication and data security has been an imperative requirement for wireless
networks throughout their evolution. The physical-layer security (PLS) concept
paved the way to catering to the security needs in wireless networks in a
sustainable way while guaranteeing theoretically secure transmissions,
independently of the computational capacity of adversaries. Therefore, it is of
paramount importance to consider a balanced trade-off between communication
reliability, sensing, and security in future networks, such as the 5G and
beyond, and the 6G. In this paper, we provide a comprehensive and system-wise
review of designed secure ISAC systems from a PLS point of view. In particular,
the impact of various physical-layer techniques, schemes, and wireless
technologies to ensure the sensing-security trade-off is studied from the
surveyed work. Furthermore, the amalgamation of PLS and ISAC is analyzed in a
broader impact by considering attacks targeting data confidentiality,
communication covertness, and sensing spoofing. The paper also serves as a
tutorial by presenting several theoretical foundations on ISAC and PLS, which
represent a practical guide for readers to develop novel secure ISAC network
designs.

### 3. [X-Driver: Explainable Autonomous Driving with Vision-Language Models](http://arxiv.org/pdf/2505.05098v1)

Authors: Wei Liu, Jiyuan Zhang, Binxiong Zheng, Yufeng Hu, Yingzhan Lin, Zengfeng Zeng

End-to-end autonomous driving has advanced significantly, offering benefits
such as system simplicity and stronger driving performance in both open-loop
and closed-loop settings than conventional pipelines. However, existing
frameworks still suffer from low success rates in closed-loop evaluations,
highlighting their limitations in real-world deployment. In this paper, we
introduce X-Driver, a unified multi-modal large language models(MLLMs)
framework designed for closed-loop autonomous driving, leveraging
Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and
decision-making. We validate X-Driver across multiple autonomous driving tasks
using public benchmarks in CARLA simulation environment, including
Bench2Drive[6]. Our experimental results demonstrate superior closed-loop
performance, surpassing the current state-of-the-art(SOTA) while improving the
interpretability of driving decisions. These findings underscore the importance
of structured reasoning in end-to-end driving and establish X-Driver as a
strong baseline for future research in closed-loop autonomous driving.

### Graphics

### 1. [Improving Global Motion Estimation in Sparse IMU-based Motion Capture with Physics](http://arxiv.org/pdf/2505.05010v1)

Authors: Xinyu Yi, Shaohua Pan, Feng Xu

By learning human motion priors, motion capture can be achieved by 6 inertial
measurement units (IMUs) in recent years with the development of deep learning
techniques, even though the sensor inputs are sparse and noisy. However, human
global motions are still challenging to be reconstructed by IMUs. This paper
aims to solve this problem by involving physics. It proposes a physical
optimization scheme based on multiple contacts to enable physically plausible
translation estimation in the full 3D space where the z-directional motion is
usually challenging for previous works. It also considers gravity in local pose
estimation which well constrains human global orientations and refines local
pose estimation in a joint estimation manner. Experiments demonstrate that our
method achieves more accurate motion capture for both local poses and global
motions. Furthermore, by deeply integrating physics, we can also estimate 3D
contact, contact forces, joint torques, and interacting proxy surfaces.

### 2. [ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators](http://arxiv.org/pdf/2505.04961v1)

Authors: Ziyu Zhang, Sergey Bashkirov, Dun Yang, Michael Taylor, Xue Bin Peng

Multi-objective optimization problems, which require the simultaneous
optimization of multiple terms, are prevalent across numerous applications.
Existing multi-objective optimization methods often rely on manually tuned
aggregation functions to formulate a joint optimization target. The performance
of such hand-tuned methods is heavily dependent on careful weight selection, a
time-consuming and laborious process. These limitations also arise in the
setting of reinforcement-learning-based motion tracking for physically
simulated characters, where intricately crafted reward functions are typically
used to achieve high-fidelity results. Such solutions not only require domain
expertise and significant manual adjustment, but also limit the applicability
of the resulting reward function across diverse skills. To bridge this gap, we
present a novel adversarial multi-objective optimization technique that is
broadly applicable to a range of multi-objective optimization problems,
including motion tracking. The proposed adversarial differential discriminator
receives a single positive sample, yet is still effective at guiding the
optimization process. We demonstrate that our technique can enable characters
to closely replicate a variety of acrobatic and agile behaviors, achieving
comparable quality to state-of-the-art motion-tracking methods, without relying
on manually tuned reward functions. Results are best visualized through
https://youtu.be/rz8BYCE9E2w.

### 3. [Inter-Diffusion Generation Model of Speakers and Listeners for Effective Communication](http://arxiv.org/pdf/2505.04996v1)

Authors: Jinhe Huang, Yongkang Cheng, Yuming Hang, Gaoge Han, Jinewei Li, Jing Zhang, Xingjian Gu

Full-body gestures play a pivotal role in natural interactions and are
crucial for achieving effective communication. Nevertheless, most existing
studies primarily focus on the gesture generation of speakers, overlooking the
vital role of listeners in the interaction process and failing to fully explore
the dynamic interaction between them. This paper innovatively proposes an
Inter-Diffusion Generation Model of Speakers and Listeners for Effective
Communication. For the first time, we integrate the full-body gestures of
listeners into the generation framework. By devising a novel inter-diffusion
mechanism, this model can accurately capture the complex interaction patterns
between speakers and listeners during communication. In the model construction
process, based on the advanced diffusion model architecture, we innovatively
introduce interaction conditions and the GAN model to increase the denoising
step size. As a result, when generating gesture sequences, the model can not
only dynamically generate based on the speaker's speech information but also
respond in realtime to the listener's feedback, enabling synergistic
interaction between the two. Abundant experimental results demonstrate that
compared with the current state-of-the-art gesture generation methods, the
model we proposed has achieved remarkable improvements in the naturalness,
coherence, and speech-gesture synchronization of the generated gestures. In the
subjective evaluation experiments, users highly praised the generated
interaction scenarios, believing that they are closer to real life human
communication situations. Objective index evaluations also show that our model
outperforms the baseline methods in multiple key indicators, providing more
powerful support for effective communication.

### 4. [An Active Contour Model for Silhouette Vectorization using Bézier Curves](http://arxiv.org/pdf/2505.05132v1)

Authors: Luis Alvarez, Jean-Michel Morel

In this paper, we propose an active contour model for silhouette
vectorization using cubic B\'ezier curves. Among the end points of the B\'ezier
curves, we distinguish between corner and regular points where the orientation
of the tangent vector is prescribed. By minimizing the distance of the B\'ezier
curves to the silhouette boundary, the active contour model optimizes the
location of the B\'ezier curves end points, the orientation of the tangent
vectors in the regular points, and the estimation of the B\'ezier curve
parameters. This active contour model can use the silhouette vectorization
obtained by any method as an initial guess. The proposed method significantly
reduces the average distance between the silhouette boundary and its
vectorization obtained by the world-class graphic software Inkscape, Adobe
Illustrator, and a curvature-based vectorization method, which we introduce for
comparison. Our method also allows us to impose additional regularity on the
B\'ezier curves by reducing their lengths.

### 5. [Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields](http://arxiv.org/pdf/2505.05356v1)

Authors: Runfeng Li, Mikhail Okunev, Zixuan Guo, Anh Ha Duong, Christian Richardt, Matthew O'Toole, James Tompkin

We present a method to reconstruct dynamic scenes from monocular
continuous-wave time-of-flight (C-ToF) cameras using raw sensor samples that
achieves similar or better accuracy than neural volumetric approaches and is
100x faster. Quickly achieving high-fidelity dynamic 3D reconstruction from a
single viewpoint is a significant challenge in computer vision. In C-ToF
radiance field reconstruction, the property of interest-depth-is not directly
measured, causing an additional challenge. This problem has a large and
underappreciated impact upon the optimization when using a fast primitive-based
scene representation like 3D Gaussian splatting, which is commonly used with
multi-view data to produce satisfactory results and is brittle in its
optimization otherwise. We incorporate two heuristics into the optimization to
improve the accuracy of scene geometry represented by Gaussians. Experimental
results show that our approach produces accurate reconstructions under
constrained C-ToF sensing conditions, including for fast motions like swinging
baseball bats. https://visual.cs.brown.edu/gftorf

### Computer Science and Game Theory

### 1. [Sample Complexity of Identifying the Nonredundancy of Nontransitive Games in Dueling Bandits](http://arxiv.org/pdf/2505.05014v1)

Authors: Shang Lu, Shuji Kijima

Dueling bandit is a variant of the Multi-armed bandit to learn the binary
relation by comparisons. Most work on the dueling bandit has targeted
transitive relations, that is, totally/partially ordered sets, or assumed at
least the existence of a champion such as Condorcet winner and Copeland winner.
This work develops an analysis of dueling bandits for non-transitive relations.
Jan-ken (a.k.a. rock-paper-scissors) is a typical example of a non-transitive
relation. It is known that a rational player chooses one of three items
uniformly at random, which is known to be Nash equilibrium in game theory.
Interestingly, any variant of Jan-ken with four items (e.g., rock, paper,
scissors, and well) contains at least one useless item, which is never selected
by a rational player. This work investigates a dueling bandit problem to
identify whether all $n$ items are indispensable in a given win-lose relation.
Then, we provide upper and lower bounds of the sample complexity of the
identification problem in terms of the determinant of $A$ and a solution of
$\mathbf{x}^{\top} A = \mathbf{0}^{\top}$ where $A$ is an $n \times n$ pay-off
matrix that every duel follows.

### 2. [Weighted Envy-Freeness Revisited: Indivisible Resource and House Allocations](http://arxiv.org/pdf/2505.05353v1)

Authors: Yuxi Liu, Mingyu Xiao

Envy-Freeness is one of the most fundamental and important concepts in fair
allocation. Some recent studies have focused on the concept of weighted
envy-freeness. Under this concept, each agent is assigned a weight, and their
valuations are divided by their weights when assessing fairness. This concept
can promote more fairness in some scenarios. But on the other hand,
experimental research has shown that this weighted envy-freeness significantly
reduces the likelihood of fair allocations. When we must allocate the
resources, we may propose fairness concepts with lower requirements that are
potentially more feasible to implement. In this paper, we revisit weighted
envy-freeness and propose a new concept called SumAvg-envy-freeness, which
substantially increases the existence of fair allocations. This new concept can
be seen as a complement of the normal weighted envy-fairness. Furthermore, we
systematically study the computational complexity of finding fair allocations
under the old and new weighted fairness concepts in two types of classic
problems: Indivisible Resource Allocation and House Allocation. Our study
provides a comprehensive characterization of various properties of weighted
envy-freeness.

### 3. [Incentive-Aware Machine Learning; Robustness, Fairness, Improvement & Causality](http://arxiv.org/pdf/2505.05211v1)

Authors: Chara Podimata

The article explores the emerging domain of incentive-aware machine learning
(ML), which focuses on algorithmic decision-making in contexts where
individuals can strategically modify their inputs to influence outcomes. It
categorizes the research into three perspectives: robustness, aiming to design
models resilient to "gaming"; fairness, analyzing the societal impacts of such
systems; and improvement/causality, recognizing situations where strategic
actions lead to genuine personal or societal improvement. The paper introduces
a unified framework encapsulating models for these perspectives, including
offline, online, and causal settings, and highlights key challenges such as
differentiating between gaming and improvement and addressing heterogeneity
among agents. By synthesizing findings from diverse works, we outline
theoretical advancements and practical solutions for robust, fair, and
causally-informed incentive-aware ML systems.

### 4. [SUUM: Timestamp-based Nakamoto-style Blockchains are Vulnerable](http://arxiv.org/pdf/2505.05328v1)

Authors: Junjie Hu, Na Ruan

We introduce two advanced attack strategies, the Unrestricted Uncle Maker
(UUM) Attack and the Staircase-Unrestricted Uncle Maker (SUUM) Attack, which
fundamentally threaten the security of timestamp-based Nakamoto-style
blockchains by inflicting permanent systemic harm. Unlike prior work that
merely enhances adversarial rewards, these attacks exploit vulnerabilities in
timestamp manipulation and fork selection rules to irreversibly destabilize
blockchain fairness and incentive mechanisms. Specifically, the SUUM attack
enables adversaries to persistently launch attacks at zero cost, eliminating
constraints on block withholding and risk-free conditions, while systematically
maximizing rewards through coordinated timestamp adjustments and strategic
block release.
  Our analysis demonstrates that SUUM adversaries achieve disproportionate
reward advantages over both UUM and the original Riskless Uncle Maker (RUM)
Attack [CCS '23], with all three strategies surpassing honest mining.
Crucially, SUUM's cost-free persistence allows adversaries to indefinitely
drain rewards from honest participants by maintaining minimal difficulty risks
through precise timestamp manipulation. This creates a self-reinforcing cycle:
adversaries amplify their profits while suppressing honest returns, thereby
permanently eroding the protocol's security assumptions. Through rigorous
theoretical modeling and simulations, we validate how SUUM's combination of
timestamp tampering, block withholding, and difficulty risk control enables
unmitigated exploitation of consensus mechanisms. This work underscores the
existential risks posed by timestamp-based Nakamoto-style protocols and
advocates urgent countermeasures to ensure long-term stability.

### 5. [Robust Online Learning with Private Information](http://arxiv.org/pdf/2505.05341v1)

Authors: Kyohei Okumura

This paper investigates the robustness of online learning algorithms when
learners possess private information. No-external-regret algorithms, prevalent
in machine learning, are vulnerable to strategic manipulation, allowing an
adaptive opponent to extract full surplus. Even standard
no-weak-external-regret algorithms, designed for optimal learning in stationary
environments, exhibit similar vulnerabilities. This raises a fundamental
question: can a learner simultaneously prevent full surplus extraction by
adaptive opponents while maintaining optimal performance in well-behaved
environments? To address this, we model the problem as a two-player repeated
game, where the learner with private information plays against the environment,
facing ambiguity about the environment's types: stationary or adaptive. We
introduce \emph{partial safety} as a key design criterion for online learning
algorithms to prevent full surplus extraction. We then propose the
\emph{Explore-Exploit-Punish} (\textsf{EEP}) algorithm and prove that it
satisfies partial safety while achieving optimal learning in stationary
environments, and has a variant that delivers improved welfare performance. Our
findings highlight the risks of applying standard online learning algorithms in
strategic settings with adverse selection. We advocate for a shift toward
online learning algorithms that explicitly incorporate safeguards against
strategic manipulation while ensuring strong learning performance.

### Human-Computer Interaction

### 1. [From First Draft to Final Insight: A Multi-Agent Approach for Feedback Generation](http://arxiv.org/pdf/2505.04869v1)

Authors: Jie Cao, Chloe Qianhui Zhao, Xian Chen, Shuman Wang, Christian Schunn, Kenneth R. Koedinger, Jionghao Lin

Producing large volumes of high-quality, timely feedback poses significant
challenges to instructors. To address this issue, automation
technologies-particularly Large Language Models (LLMs)-show great potential.
However, current LLM-based research still shows room for improvement in terms
of feedback quality. Our study proposed a multi-agent approach performing
"generation, evaluation, and regeneration" (G-E-RG) to further enhance feedback
quality. In the first-generation phase, six methods were adopted, combining
three feedback theoretical frameworks and two prompt methods: zero-shot and
retrieval-augmented generation with chain-of-thought (RAG_CoT). The results
indicated that, compared to first-round feedback, G-E-RG significantly improved
final feedback across six methods for most dimensions. Specifically:(1)
Evaluation accuracy for six methods increased by 3.36% to 12.98% (p<0.001); (2)
The proportion of feedback containing four effective components rose from an
average of 27.72% to an average of 98.49% among six methods, sub-dimensions of
providing critiques, highlighting strengths, encouraging agency, and
cultivating dialogue also showed great enhancement (p<0.001); (3) There was a
significant improvement in most of the feature values (p<0.001), although some
sub-dimensions (e.g., strengthening the teacher-student relationship) still
require further enhancement; (4) The simplicity of feedback was effectively
enhanced (p<0.001) for three methods.

### 2. [Theatrical Language Processing: Exploring AI-Augmented Improvisational Acting and Scriptwriting with LLMs](http://arxiv.org/pdf/2505.04890v1)

Authors: Sora Kang, Joonhwan Lee

The increasing convergence of artificial intelligence has opened new avenues,
including its emerging role in enhancing creativity. It is reshaping
traditional creative practices such as actor improvisation, which often
struggles with predictable patterns, limited interaction, and a lack of
engaging stimuli. In this paper, we introduce a new concept, Theatrical
Language Processing (TLP), and an AI-driven creativity support tool,
Scribble.ai, designed to augment actors' creative expression and spontaneity
through interactive practice. We conducted a user study involving tests and
interviews with fourteen participants. Our findings indicate that: (1) Actors
expanded their creativity when faced with AI-produced irregular scenarios; (2)
The AI's unpredictability heightened their problem-solving skills, specifically
in interpreting unfamiliar situations; (3) However, AI often generated
excessively detailed scripts, which limited interpretive freedom and hindered
subtext exploration. Based on these findings, we discuss the new potential in
enhancing creative expressions in film and theater studies through an AI-driven
tool.

### 3. [Uncertainty-Aware Scarf Plots](http://arxiv.org/pdf/2505.05038v1)

Authors: Nelusa Pathmanathan, Seyda Öney, Maurice Koch, Daniel Weiskopf, Kuno Kurzhals

Multiple challenges emerge when analyzing eye-tracking data with areas of
interest (AOIs) because recordings are subject to different sources of
uncertainties. Previous work often presents gaze data without considering those
inaccuracies in the data. To address this issue, we developed uncertainty-aware
scarf plot visualizations that aim to make analysts aware of uncertainties with
respect to the position-based mapping of gaze to AOIs and depth dependency in
3D scenes. Additionally, we also consider uncertainties in automatic AOI
annotation. We showcase our approach in comparison to standard scarf plots in
an augmented reality scenario.

### 4. [GesPrompt: Leveraging Co-Speech Gestures to Augment LLM-Based Interaction in Virtual Reality](http://arxiv.org/pdf/2505.05441v1)

Authors: Xiyun Hu, Dizhi Ma, Fengming He, Zhengzhe Zhu, Shao-Kang Hsia, Chenfei Zhu, Ziyi Liu, Karthik Ramani

Large Language Model (LLM)-based copilots have shown great potential in
Extended Reality (XR) applications. However, the user faces challenges when
describing the 3D environments to the copilots due to the complexity of
conveying spatial-temporal information through text or speech alone. To address
this, we introduce GesPrompt, a multimodal XR interface that combines co-speech
gestures with speech, allowing end-users to communicate more naturally and
accurately with LLM-based copilots in XR environments. By incorporating
gestures, GesPrompt extracts spatial-temporal reference from co-speech
gestures, reducing the need for precise textual prompts and minimizing
cognitive load for end-users. Our contributions include (1) a workflow to
integrate gesture and speech input in the XR environment, (2) a prototype VR
system that implements the workflow, and (3) a user study demonstrating its
effectiveness in improving user communication in VR environments.

### 5. [Fairness Perceptions in Regression-based Predictive Models](http://arxiv.org/pdf/2505.04886v1)

Authors: Mukund Telukunta, Venkata Sriram Siddhardh Nadendla, Morgan Stuart, Casey Canfield

Regression-based predictive analytics used in modern kidney transplantation
is known to inherit biases from training data. This leads to social
discrimination and inefficient organ utilization, particularly in the context
of a few social groups. Despite this concern, there is limited research on
fairness in regression and its impact on organ utilization and placement. This
paper introduces three novel divergence-based group fairness notions: (i)
independence, (ii) separation, and (iii) sufficiency to assess the fairness of
regression-based analytics tools. In addition, fairness preferences are
investigated from crowd feedback, in order to identify a socially accepted
group fairness criterion for evaluating these tools. A total of 85 participants
were recruited from the Prolific crowdsourcing platform, and a Mixed-Logit
discrete choice model was used to model fairness feedback and estimate social
fairness preferences. The findings clearly depict a strong preference towards
the separation and sufficiency fairness notions, and that the predictive
analytics is deemed fair with respect to gender and race groups, but unfair in
terms of age groups.

### 6. [Dukawalla: Voice Interfaces for Small Businesses in Africa](http://arxiv.org/pdf/2505.05170v1)

Authors: Elizabeth Ankrah, Stephanie Nyairo, Mercy Muchai, Kagonya Awori, Millicent Ochieng, Mark Kariuki, Jacki O'Neill

Small and medium sized businesses often struggle with data driven decision
making do to a lack of advanced analytics tools, especially in African
countries where they make up a majority of the workforce. Though many tools
exist they are not designed to fit into the ways of working of SMB workers who
are mobile first, have limited time to learn new workflows, and for whom social
and business are tightly coupled. To address this, the Dukawalla prototype was
created. This intelligent assistant bridges the gap between raw business data,
and actionable insights by leveraging voice interaction and the power of
generative AI. Dukawalla provides an intuitive way for business owners to
interact with their data, aiding in informed decision making. This paper
examines Dukawalla's deployment across SMBs in Nairobi, focusing on their
experiences using this voice based assistant to streamline data collection and
provide business insights

### 7. [A Pain Assessment Framework based on multimodal data and Deep Machine Learning methods](http://arxiv.org/pdf/2505.05396v1)

Authors: Stefanos Gkikas

From the original abstract:
  This thesis initially aims to study the pain assessment process from a
clinical-theoretical perspective while exploring and examining existing
automatic approaches. Building on this foundation, the primary objective of
this Ph.D. project is to develop innovative computational methods for automatic
pain assessment that achieve high performance and are applicable in real
clinical settings. A primary goal is to thoroughly investigate and assess
significant factors, including demographic elements that impact pain
perception, as recognized in pain research, through a computational standpoint.
Within the limits of the available data in this research area, our goal was to
design, develop, propose, and offer automatic pain assessment pipelines for
unimodal and multimodal configurations that are applicable to the specific
requirements of different scenarios. The studies published in this Ph.D. thesis
showcased the effectiveness of the proposed methods, achieving state-of-the-art
results. Additionally, they paved the way for exploring new approaches in
artificial intelligence, foundation models, and generative artificial
intelligence.

### 8. [A Multi-Agent AI Framework for Immersive Audiobook Production through Spatial Audio and Neural Narration](http://arxiv.org/pdf/2505.04885v1)

Authors: Shaja Arul Selvamani, Nia D'Souza Ganapathy

This research introduces an innovative AI-driven multi-agent framework
specifically designed for creating immersive audiobooks. Leveraging neural
text-to-speech synthesis with FastSpeech 2 and VALL-E for expressive narration
and character-specific voices, the framework employs advanced language models
to automatically interpret textual narratives and generate realistic spatial
audio effects. These sound effects are dynamically synchronized with the
storyline through sophisticated temporal integration methods, including Dynamic
Time Warping (DTW) and recurrent neural networks (RNNs). Diffusion-based
generative models combined with higher-order ambisonics (HOA) and scattering
delay networks (SDN) enable highly realistic 3D soundscapes, substantially
enhancing listener immersion and narrative realism. This technology
significantly advances audiobook applications, providing richer experiences for
educational content, storytelling platforms, and accessibility solutions for
visually impaired audiences. Future work will address personalization, ethical
management of synthesized voices, and integration with multi-sensory platforms.

### 9. [Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects](http://arxiv.org/pdf/2505.05318v1)

Authors: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci

The rapid adoption of Vision Language Models (VLMs), pre-trained on large
image-text and video-text datasets, calls for protecting and informing users
about when to trust these systems. This survey reviews studies on trust
dynamics in user-VLM interactions, through a multi-disciplinary taxonomy
encompassing different cognitive science capabilities, collaboration modes, and
agent behaviours. Literature insights and findings from a workshop with
prospective VLM users inform preliminary requirements for future VLM trust
studies.

### Information Retrieval

### 1. [LSRP: A Leader-Subordinate Retrieval Framework for Privacy-Preserving Cloud-Device Collaboration](http://arxiv.org/pdf/2505.05031v1)

Authors: Yingyi Zhang, Pengyue Jia, Xianneng Li, Derong Xu, Maolin Wang, Yichao Wang, Zhaocheng Du, Huifeng Guo, Yong Liu, Ruiming Tang, Xiangyu Zhao

Cloud-device collaboration leverages on-cloud Large Language Models (LLMs)
for handling public user queries and on-device Small Language Models (SLMs) for
processing private user data, collectively forming a powerful and
privacy-preserving solution. However, existing approaches often fail to fully
leverage the scalable problem-solving capabilities of on-cloud LLMs while
underutilizing the advantage of on-device SLMs in accessing and processing
personalized data. This leads to two interconnected issues: 1) Limited
utilization of the problem-solving capabilities of on-cloud LLMs, which fail to
align with personalized user-task needs, and 2) Inadequate integration of user
data into on-device SLM responses, resulting in mismatches in contextual user
information.
  In this paper, we propose a Leader-Subordinate Retrieval framework for
Privacy-preserving cloud-device collaboration (LSRP), a novel solution that
bridges these gaps by: 1) enhancing on-cloud LLM guidance to on-device SLM
through a dynamic selection of task-specific leader strategies named as
user-to-user retrieval-augmented generation (U-U-RAG), and 2) integrating the
data advantages of on-device SLMs through small model feedback Direct
Preference Optimization (SMFB-DPO) for aligning the on-cloud LLM with the
on-device SLM. Experiments on two datasets demonstrate that LSRP consistently
outperforms state-of-the-art baselines, significantly improving question-answer
relevance and personalization, while preserving user privacy through efficient
on-device retrieval. Our code is available at:
https://github.com/Zhang-Yingyi/LSRP.

### 2. [Divide-and-Conquer: Cold-Start Bundle Recommendation via Mixture of Diffusion Experts](http://arxiv.org/pdf/2505.05035v1)

Authors: Ming Li, Lin Li, Xiaohui Tao, Dong Zhang, Jimmy Xiangji Huang

Cold-start bundle recommendation focuses on modeling new bundles with
insufficient information to provide recommendations. Advanced bundle
recommendation models usually learn bundle representations from multiple views
(e.g., interaction view) at both the bundle and item levels. Consequently, the
cold-start problem for bundles is more challenging than that for traditional
items due to the dual-level multi-view complexity. In this paper, we propose a
novel Mixture of Diffusion Experts (MoDiffE) framework, which employs a
divide-and-conquer strategy for cold-start bundle recommendation and follows
three steps:(1) Divide: The bundle cold-start problem is divided into
independent but similar sub-problems sequentially by level and view, which can
be summarized as the poor representation of feature-missing bundles in
prior-embedding models. (2) Conquer: Beyond prior-embedding models that
fundamentally provide the embedded representations, we introduce a
diffusion-based method to solve all sub-problems in a unified way, which
directly generates diffusion representations using diffusion models without
depending on specific features. (3) Combine: A cold-aware hierarchical Mixture
of Experts (MoE) is employed to combine results of the sub-problems for final
recommendations, where the two models for each view serve as experts and are
adaptively fused for different bundles in a multi-layer manner. Additionally,
MoDiffE adopts a multi-stage decoupled training pipeline and introduces a
cold-start gating augmentation method to enable the training of gating for cold
bundles. Through extensive experiments on three real-world datasets, we
demonstrate that MoDiffE significantly outperforms existing solutions in
handling cold-start bundle recommendation. It achieves up to a 0.1027 absolute
gain in Recall@20 in cold-start scenarios and up to a 47.43\% relative
improvement in all-bundle scenarios.

### 3. [Hybrid Personalization Using Declarative and Procedural Memory Modules of the Cognitive Architecture ACT-R](http://arxiv.org/pdf/2505.05083v1)

Authors: Kevin Innerebner, Dominik Kowald, Markus Schedl, Elisabeth Lex

Recommender systems often rely on sub-symbolic machine learning approaches
that operate as opaque black boxes. These approaches typically fail to account
for the cognitive processes that shape user preferences and decision-making. In
this vision paper, we propose a hybrid user modeling framework based on the
cognitive architecture ACT-R that integrates symbolic and sub-symbolic
representations of human memory. Our goal is to combine ACT-R's declarative
memory, which is responsible for storing symbolic chunks along sub-symbolic
activations, with its procedural memory, which contains symbolic production
rules. This integration will help simulate how users retrieve past experiences
and apply decision-making strategies. With this approach, we aim to provide
more transparent recommendations, enable rule-based explanations, and
facilitate the modeling of cognitive biases. We argue that our approach has the
potential to inform the design of a new generation of human-centered,
psychology-informed recommender systems.

### 4. [Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems](http://arxiv.org/pdf/2505.05196v1)

Authors: Fatemeh Nazary, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio

We present a systematic study of provider-side data poisoning in
retrieval-augmented recommender systems (RAG-based). By modifying only a small
fraction of tokens within item descriptions -- for instance, adding emotional
keywords or borrowing phrases from semantically related items -- an attacker
can significantly promote or demote targeted items. We formalize these attacks
under token-edit and semantic-similarity constraints, and we examine their
effectiveness in both promotion (long-tail items) and demotion (short-head
items) scenarios. Our experiments on MovieLens, using two large language model
(LLM) retrieval modules, show that even subtle attacks shift final rankings and
item exposures while eluding naive detection. The results underscore the
vulnerability of RAG-based pipelines to small-scale metadata rewrites and
emphasize the need for robust textual consistency checks and provenance
tracking to thwart stealthy provider-side poisoning.

### 5. [Artifact Sharing for Information Retrieval Research](http://arxiv.org/pdf/2505.05434v1)

Authors: Sean MacAvaney

Sharing artifacts -- such as trained models, pre-built indexes, and the code
to use them -- aids in reproducibility efforts by allowing researchers to
validate intermediate steps and improves the sustainability of research by
allowing multiple groups to build off one another's prior computational work.
Although there are de facto consensuses on how to share research code (through
a git repository linked to from publications) and trained models (via
HuggingFace Hub), there is no consensus for other types of artifacts, such as
built indexes. Given the practical utility of using shared indexes, researchers
have resorted to self-hosting these resources or performing ad hoc file
transfers upon request, ultimately limiting the artifacts' discoverability and
reuse. This demonstration introduces a flexible and interoperable way to share
artifacts for Information Retrieval research, improving both their
accessibility and usability.

### 6. [QBR: A Question-Bank-Based Approach to Fine-Grained Legal Knowledge Retrieval for the General Public](http://arxiv.org/pdf/2505.04883v1)

Authors: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu

Retrieval of legal knowledge by the general public is a challenging problem
due to the technicality of the professional knowledge and the lack of
fundamental understanding by laypersons on the subject. Traditional information
retrieval techniques assume that users are capable of formulating succinct and
precise queries for effective document retrieval. In practice, however, the
wide gap between the highly technical contents and untrained users makes legal
knowledge retrieval very difficult. We propose a methodology, called QBR, which
employs a Questions Bank (QB) as an effective medium for bridging the knowledge
gap. We show how the QB is used to derive training samples to enhance the
embedding of knowledge units within documents, which leads to effective
fine-grained knowledge retrieval. We discuss and evaluate through experiments
various advantages of QBR over traditional methods. These include more
accurate, efficient, and explainable document retrieval, better comprehension
of retrieval results, and highly effective fine-grained knowledge retrieval. We
also present some case studies and show that QBR achieves social impact by
assisting citizens to resolve everyday legal concerns.

### 7. [FF-PNet: A Pyramid Network Based on Feature and Field for Brain Image Registration](http://arxiv.org/pdf/2505.04938v1)

Authors: Ying Zhang, Shuai Guo, Chenxi Sun, Yuchen Zhu, Jinhai Xiang

In recent years, deformable medical image registration techniques have made
significant progress. However, existing models still lack efficiency in
parallel extraction of coarse and fine-grained features. To address this, we
construct a new pyramid registration network based on feature and deformation
field (FF-PNet). For coarse-grained feature extraction, we design a Residual
Feature Fusion Module (RFFM), for fine-grained image deformation, we propose a
Residual Deformation Field Fusion Module (RDFFM). Through the parallel
operation of these two modules, the model can effectively handle complex image
deformations. It is worth emphasizing that the encoding stage of FF-PNet only
employs traditional convolutional neural networks without any attention
mechanisms or multilayer perceptrons, yet it still achieves remarkable
improvements in registration accuracy, fully demonstrating the superior feature
decoding capabilities of RFFM and RDFFM. We conducted extensive experiments on
the LPBA and OASIS datasets. The results show our network consistently
outperforms popular methods in metrics like the Dice Similarity Coefficient.

### 8. [Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations](http://arxiv.org/pdf/2505.04948v1)

Authors: Md Aminul Islam, Ahmed Sayeed Faruk

Recommender systems are essential for delivering personalized content across
digital platforms by modeling user preferences and behaviors. Recently, large
language models (LLMs) have been adopted for prompt-based recommendation due to
their ability to generate personalized outputs without task-specific training.
However, LLM-based methods face limitations such as limited context window
size, inefficient pointwise and pairwise prompting, and difficulty handling
listwise ranking due to token constraints. LLMs can also be sensitive to
position bias, as they may overemphasize earlier items in the prompt regardless
of their true relevance. To address and investigate these issues, we propose a
hybrid framework that combines a traditional recommendation model with an LLM
for reranking top-k items using structured prompts. We evaluate the effects of
user history reordering and instructional prompts for mitigating position bias.
Experiments on MovieLens-100K show that randomizing user history improves
ranking quality, but LLM-based reranking does not outperform the base model.
Explicit instructions to reduce position bias are also ineffective. Our
evaluations reveal limitations in LLMs' ability to model ranking context and
mitigate bias. Our code is publicly available at
https://github.com/aminul7506/LLMForReRanking.

### 9. [Learning Item Representations Directly from Multimodal Features for Effective Recommendation](http://arxiv.org/pdf/2505.04960v1)

Authors: Xin Zhou, Xiaoxiong Zhang, Dusit Niyato, Zhiqi Shen

Conventional multimodal recommender systems predominantly leverage Bayesian
Personalized Ranking (BPR) optimization to learn item representations by
amalgamating item identity (ID) embeddings with multimodal features.
Nevertheless, our empirical and theoretical findings unequivocally demonstrate
a pronounced optimization gradient bias in favor of acquiring representations
from multimodal features over item ID embeddings. As a consequence, item ID
embeddings frequently exhibit suboptimal characteristics despite the
convergence of multimodal feature parameters. Given the rich informational
content inherent in multimodal features, in this paper, we propose a novel
model (i.e., LIRDRec) that learns item representations directly from these
features to augment recommendation performance. Recognizing that features
derived from each modality may capture disparate yet correlated aspects of
items, we propose a multimodal transformation mechanism, integrated with
modality-specific encoders, to effectively fuse features from all modalities.
Moreover, to differentiate the influence of diverse modality types, we devise a
progressive weight copying fusion module within LIRDRec. This module
incrementally learns the weight assigned to each modality in synthesizing the
final user or item representations. Finally, we utilize the powerful visual
understanding of Multimodal Large Language Models (MLLMs) to convert the item
images into texts and extract semantics embeddings upon the texts via LLMs.
Empirical evaluations conducted on five real-world datasets validate the
superiority of our approach relative to competing baselines. It is worth noting
the proposed model, equipped with embeddings extracted from MLLMs and LLMs, can
further improve the recommendation accuracy of NDCG@20 by an average of 4.21%
compared to the original embeddings.

### 10. [The Pitfalls of Growing Group Complexity: LLMs and Social Choice-Based Aggregation for Group Recommendations](http://arxiv.org/pdf/2505.05016v1)

Authors: Cedric Waterschoot, Nava Tintarev, Francesco Barile

Large Language Models (LLMs) are increasingly applied in recommender systems
aimed at both individuals and groups. Previously, Group Recommender Systems
(GRS) often used social choice-based aggregation strategies to derive a single
recommendation based on the preferences of multiple people. In this paper, we
investigate under which conditions language models can perform these strategies
correctly based on zero-shot learning and analyse whether the formatting of the
group scenario in the prompt affects accuracy. We specifically focused on the
impact of group complexity (number of users and items), different LLMs,
different prompting conditions, including In-Context learning or generating
explanations, and the formatting of group preferences. Our results show that
performance starts to deteriorate when considering more than 100 ratings.
However, not all language models were equally sensitive to growing group
complexity. Additionally, we showed that In-Context Learning (ICL) can
significantly increase the performance at higher degrees of group complexity,
while adding other prompt modifications, specifying domain cues or prompting
for explanations, did not impact accuracy. We conclude that future research
should include group complexity as a factor in GRS evaluation due to its effect
on LLM performance. Furthermore, we showed that formatting the group scenarios
differently, such as rating lists per user or per item, affected accuracy. All
in all, our study implies that smaller LLMs are capable of generating group
recommendations under the right conditions, making the case for using smaller
models that require less computing power and costs.

### Machine Learning

### 1. [GCN-Based Throughput-Oriented Handover Management in Dense 5G Vehicular Networks](http://arxiv.org/pdf/2505.04894v1)

Authors: Nazanin Mehregan, Robson E. De Grande

The rapid advancement of 5G has transformed vehicular networks, offering high
bandwidth, low latency, and fast data rates essential for real-time
applications in smart cities and vehicles. These improvements enhance traffic
safety and entertainment services. However, the limited coverage and frequent
handovers in 5G networks cause network instability, especially in high-mobility
environments due to the ping-pong effect. This paper presents TH-GCN
(Throughput-oriented Graph Convolutional Network), a novel approach for
optimizing handover management in dense 5G networks. Using graph neural
networks (GNNs), TH-GCN models vehicles and base stations as nodes in a dynamic
graph enriched with features such as signal quality, throughput, vehicle speed,
and base station load. By integrating both user equipment and base station
perspectives, this dual-centric approach enables adaptive, real-time handover
decisions that improve network stability. Simulation results show that TH-GCN
reduces handovers by up to 78 percent and improves signal quality by 10
percent, outperforming existing methods.

### 2. [VaCDA: Variational Contrastive Alignment-based Scalable Human Activity Recognition](http://arxiv.org/pdf/2505.04907v1)

Authors: Soham Khisa, Avijoy Chakma

Technological advancements have led to the rise of wearable devices with
sensors that continuously monitor user activities, generating vast amounts of
unlabeled data. This data is challenging to interpret, and manual annotation is
labor-intensive and error-prone. Additionally, data distribution is often
heterogeneous due to device placement, type, and user behavior variations. As a
result, traditional transfer learning methods perform suboptimally, making it
difficult to recognize daily activities. To address these challenges, we use a
variational autoencoder (VAE) to learn a shared, low-dimensional latent space
from available sensor data. This space generalizes data across diverse sensors,
mitigating heterogeneity and aiding robust adaptation to the target domain. We
integrate contrastive learning to enhance feature representation by aligning
instances of the same class across domains while separating different classes.
We propose Variational Contrastive Domain Adaptation (VaCDA), a multi-source
domain adaptation framework combining VAEs and contrastive learning to improve
feature representation and reduce heterogeneity between source and target
domains. We evaluate VaCDA on multiple publicly available datasets across three
heterogeneity scenarios: cross-person, cross-position, and cross-device. VaCDA
outperforms the baselines in cross-position and cross-device scenarios.

### 3. [Graph Neural Network Aided Deep Reinforcement Learning for Resource Allocation in Dynamic Terahertz UAV Networks](http://arxiv.org/pdf/2505.04981v1)

Authors: Zhifeng Hu, Chong Han

Terahertz (THz) unmanned aerial vehicle (UAV) networks with flexible
topologies and ultra-high data rates are expected to empower numerous
applications in security surveillance, disaster response, and environmental
monitoring, among others. However, the dynamic topologies hinder the efficient
long-term joint power and antenna array resource allocation for THz links among
UAVs. Furthermore, the continuous nature of power and the discrete nature of
antennas cause this joint resource allocation problem to be a mixed-integer
nonlinear programming (MINLP) problem with non-convexity and NP-hardness.
Inspired by recent rapid advancements in deep reinforcement learning (DRL), a
graph neural network (GNN) aided DRL algorithm for resource allocation in the
dynamic THz UAV network with an emphasis on self-node features (GLOVE) is
proposed in this paper, with the aim of resource efficiency (RE) maximization.
When training the allocation policy for each UAV, GLOVE learns the relationship
between this UAV and its neighboring UAVs via GNN, while also emphasizing the
important self-node features of this UAV. In addition, a multi-task structure
is leveraged by GLOVE to cooperatively train resource allocation decisions for
the power and sub-arrays of all UAVs. Experimental results illustrate that
GLOVE outperforms benchmark schemes in terms of the highest RE and the lowest
latency. Moreover, unlike the benchmark methods with severe packet loss, GLOVE
maintains zero packet loss during the entire training process, demonstrating
its better robustness under the highly dynamic THz UAV network.

### 4. [Generative Models for Long Time Series: Approximately Equivariant Recurrent Network Structures for an Adjusted Training Scheme](http://arxiv.org/pdf/2505.05020v1)

Authors: Ruwen Fulek, Markus Lange-Hegermann

We present a simple yet effective generative model for time series data based
on a Variational Autoencoder (VAE) with recurrent layers, referred to as the
Recurrent Variational Autoencoder with Subsequent Training (RVAE-ST). Our
method introduces an adapted training scheme that progressively increases the
sequence length, addressing the challenge recurrent layers typically face when
modeling long sequences. By leveraging the recurrent architecture, the model
maintains a constant number of parameters regardless of sequence length. This
design encourages approximate time-shift equivariance and enables efficient
modeling of long-range temporal dependencies. Rather than introducing a
fundamentally new architecture, we show that a carefully composed combination
of known components can match or outperform state-of-the-art generative models
on several benchmark datasets. Our model performs particularly well on time
series that exhibit quasi-periodic structure,while remaining competitive on
datasets with more irregular or partially non-stationary behavior. We evaluate
its performance using ELBO, Fr\'echet Distance, discriminative scores, and
visualizations of the learned embeddings.

### 5. [Neural Pathways to Program Success: Hopfield Networks for PERT Analysis](http://arxiv.org/pdf/2505.05047v1)

Authors: Azgar Ali Noor Ahamed

Project and task scheduling under uncertainty remains a fundamental challenge
in program and project management, where accurate estimation of task durations
and dependencies is critical for delivering complex, multi project systems. The
Program Evaluation and Review Technique provides a probabilistic framework to
model task variability and critical paths. In this paper, the author presents a
novel formulation of PERT scheduling as an energy minimization problem within a
Hopfield neural network architecture. By mapping task start times and
precedence constraints into a neural computation framework, the networks
inherent optimization dynamics is exploited to approximate globally consistent
schedules. The author addresses key theoretical issues related to energy
function differentiability, constraint encoding, and convergence, and extends
the Hopfield model for structured precedence graphs. Numerical simulations on
synthetic project networks comprising up to 1000 tasks demonstrate the
viability of this approach, achieving near optimal makespans with minimal
constraint violations. The findings suggest that neural optimization models
offer a promising direction for scalable and adaptive project tasks scheduling
under uncertainty in areas such as the agentic AI workflows, microservice based
applications that the modern AI systems are being built upon.

### 6. [WaterDrum: Watermarking for Data-centric Unlearning Metric](http://arxiv.org/pdf/2505.05064v1)

Authors: Xinyang Lu, Xinyuan Niu, Gregory Kang Ruey Lau, Bui Thi Cam Nhung, Rachael Hwee Ling Sim, Fanyu Wen, Chuan-Sheng Foo, See-Kiong Ng, Bryan Kian Hsiang Low

Large language model (LLM) unlearning is critical in real-world applications
where it is necessary to efficiently remove the influence of private,
copyrighted, or harmful data from some users. However, existing utility-centric
unlearning metrics (based on model utility) may fail to accurately evaluate the
extent of unlearning in realistic settings such as when (a) the forget and
retain set have semantically similar content, (b) retraining the model from
scratch on the retain set is impractical, and/or (c) the model owner can
improve the unlearning metric without directly performing unlearning on the
LLM. This paper presents the first data-centric unlearning metric for LLMs
called WaterDrum that exploits robust text watermarking for overcoming these
limitations. We also introduce new benchmark datasets for LLM unlearning that
contain varying levels of similar data points and can be used to rigorously
evaluate unlearning algorithms using WaterDrum. Our code is available at
https://github.com/lululu008/WaterDrum and our new benchmark datasets are
released at https://huggingface.co/datasets/Glow-AI/WaterDrum-Ax.

### 7. [A Conjoint Graph Representation Learning Framework for Hypertension Comorbidity Risk Prediction](http://arxiv.org/pdf/2505.05094v1)

Authors: Leming Zhou, Zuo Wang, Zhixuan Duan

The comorbidities of hypertension impose a heavy burden on patients and
society. Early identification is necessary to prompt intervention, but it
remains a challenging task. This study aims to address this challenge by
combining joint graph learning with network analysis. Motivated by this
discovery, we develop a Conjoint Graph Representation Learning (CGRL) framework
that: a) constructs two networks based on disease coding, including the patient
network and the disease difference network. Three comorbidity network features
were generated based on the basic difference network to capture the potential
relationship between comorbidities and risk diseases; b) incorporates
computational structure intervention and learning feature representation, CGRL
was developed to predict the risks of diabetes and coronary heart disease in
patients; and c) analysis the comorbidity patterns and exploring the pathways
of disease progression, the pathological pathogenesis of diabetes and coronary
heart disease may be revealed. The results show that the network features
extracted based on the difference network are important, and the framework we
proposed provides more accurate predictions than other strong models in terms
of accuracy.

### 8. [Taming OOD Actions for Offline Reinforcement Learning: An Advantage-Based Approach](http://arxiv.org/pdf/2505.05126v1)

Authors: Xuyang Chen, Keyu Yan, Lin Zhao

Offline reinforcement learning (RL) aims to learn decision-making policies
from fixed datasets without online interactions, providing a practical solution
where online data collection is expensive or risky. However, offline RL often
suffers from distribution shift, resulting in inaccurate evaluation and
substantial overestimation on out-of-distribution (OOD) actions. To address
this, existing approaches incorporate conservatism by indiscriminately
discouraging all OOD actions, thereby hindering the agent's ability to
generalize and exploit beneficial ones. In this paper, we propose
Advantage-based Diffusion Actor-Critic (ADAC), a novel method that
systematically evaluates OOD actions using the batch-optimal value function.
Based on this evaluation, ADAC defines an advantage function to modulate the
Q-function update, enabling more precise assessment of OOD action quality. We
design a custom PointMaze environment and collect datasets to visually reveal
that advantage modulation can effectively identify and select superior OOD
actions. Extensive experiments show that ADAC achieves state-of-the-art
performance on almost all tasks in the D4RL benchmark, with particularly clear
margins on the more challenging tasks.

### 9. [Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](http://arxiv.org/pdf/2505.05143v1)

Authors: Mohammed Adnan, Rohan Jain, Ekansh Sharma, Rahul Krishnan, Yani Ioannou

The Lottery Ticket Hypothesis (LTH) suggests there exists a sparse LTH mask
and weights that achieve the same generalization performance as the dense model
while using significantly fewer parameters. However, finding a LTH solution is
computationally expensive, and a LTH sparsity mask does not generalize to other
random weight initializations. Recent work has suggested that neural networks
trained from random initialization find solutions within the same basin modulo
permutation, and proposes a method to align trained models within the same loss
basin. We hypothesize that misalignment of basins is the reason why LTH masks
do not generalize to new random initializations and propose permuting the LTH
mask to align with the new optimization basin when performing sparse training
from a different random init. We empirically show a significant increase in
generalization when sparse training from random initialization with the
permuted mask as compared to using the non-permuted LTH mask, on multiple
datasets (CIFAR-10, CIFAR-100 and ImageNet) and models (VGG11, ResNet20 and
ResNet50).

### 10. [Bandit Max-Min Fair Allocation](http://arxiv.org/pdf/2505.05169v1)

Authors: Tsubasa Harada, Shinji Ito, Hanna Sumita

In this paper, we study a new decision-making problem called the bandit
max-min fair allocation (BMMFA) problem. The goal of this problem is to
maximize the minimum utility among agents with additive valuations by
repeatedly assigning indivisible goods to them. One key feature of this problem
is that each agent's valuation for each item can only be observed through the
semi-bandit feedback, while existing work supposes that the item values are
provided at the beginning of each round. Another key feature is that the
algorithm's reward function is not additive with respect to rounds, unlike most
bandit-setting problems.
  Our first contribution is to propose an algorithm that has an asymptotic
regret bound of $O(m\sqrt{T}\ln T/n + m\sqrt{T \ln(mnT)})$, where $n$ is the
number of agents, $m$ is the number of items, and $T$ is the time horizon. This
is based on a novel combination of bandit techniques and a resource allocation
algorithm studied in the literature on competitive analysis. Our second
contribution is to provide the regret lower bound of $\Omega(m\sqrt{T}/n)$.
When $T$ is sufficiently larger than $n$, the gap between the upper and lower
bounds is a logarithmic factor of $T$.

### Neural and Evolutionary Computing

### 1. [Guiding Evolutionary AutoEncoder Training with Activation-Based Pruning Operators](http://arxiv.org/pdf/2505.05138v1)

Authors: Steven Jorgensen, Erik Hemberg, Jamal Toutouh, Una-May O'Reilly

This study explores a novel approach to neural network pruning using
evolutionary computation, focusing on simultaneously pruning the encoder and
decoder of an autoencoder. We introduce two new mutation operators that use
layer activations to guide weight pruning. Our findings reveal that one of
these activation-informed operators outperforms random pruning, resulting in
more efficient autoencoders with comparable performance to canonically trained
models. Prior work has established that autoencoder training is effective and
scalable with a spatial coevolutionary algorithm that cooperatively coevolves a
population of encoders with a population of decoders, rather than one
autoencoder. We evaluate how the same activity-guided mutation operators
transfer to this context. We find that random pruning is better than guided
pruning, in the coevolutionary setting. This suggests activation-based guidance
proves more effective in low-dimensional pruning environments, where
constrained sample spaces can lead to deviations from true uniformity in
randomization. Conversely, population-driven strategies enhance robustness by
expanding the total pruning dimensionality, achieving statistically uniform
randomness that better preserves system dynamics. We experiment with pruning
according to different schedules and present best combinations of operator and
schedule for the canonical and coevolving populations cases.

### 2. [Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks](http://arxiv.org/pdf/2505.05375v1)

Authors: Kejie Zhao, Wenjia Hua, Aiersi Tuerhong, Luziwei Leng, Yuxin Ma, Qinghua Guo

Recently, spiking neural networks (SNNs), deployed on neuromorphic chips,
provide highly efficient solutions on edge devices in different scenarios.
However, their ability to adapt to distribution shifts after deployment has
become a crucial challenge. Online test-time adaptation (OTTA) offers a
promising solution by enabling models to dynamically adjust to new data
distributions without requiring source data or labeled target samples.
Nevertheless, existing OTTA methods are largely designed for traditional
artificial neural networks and are not well-suited for SNNs. To address this
gap, we propose a low-power, neuromorphic chip-friendly online test-time
adaptation framework, aiming to enhance model generalization under distribution
shifts. The proposed approach is called Threshold Modulation (TM), which
dynamically adjusts the firing threshold through neuronal dynamics-inspired
normalization, being more compatible with neuromorphic hardware. Experimental
results on benchmark datasets demonstrate the effectiveness of this method in
improving the robustness of SNNs against distribution shifts while maintaining
low computational cost. The proposed method offers a practical solution for
online test-time adaptation of SNNs, providing inspiration for the design of
future neuromorphic chips. The demo code is available at
github.com/NneurotransmitterR/TM-OTTA-SNN.

### Networking and Internet Architecture

### 1. [Network Digital Twin for Route Optimization in 5G/B5G Transport Slicing with What-If Analysis](http://arxiv.org/pdf/2505.04879v1)

Authors: Rebecca Aben-Athar, Heitor Anglada, Lucas Costa, João Albuquerque, Abrahão Ferreira, Cristiano Bonato Both, Kleber Cardoso, Silvia Lins, Andrey Silva, Glauco Gonçalves, Ilan Correa, Aldebaro Klautau

The advent of fifth-generation (5G) and Beyond 5G (B5G) networks introduces
diverse service requirements, from ultra-low latency to high bandwidth,
demanding dynamic monitoring and advanced solutions to ensure Quality of
Service (QoS). The transport network - responsible for interconnecting the
radio access network and core networks - will increasingly face challenges in
efficiently managing complex traffic patterns. The Network Digital Twin (NDT)
concept emerges as a promising solution for testing configurations and
algorithms in a virtual network before real-world deployment. In this context,
this work designs an experimental platform with NDT in a transport network
domain, synchronizing with the virtual counterpart and a recommendation system
for what-if analysis, enabling intelligent decision-making for dynamic route
optimization problems in 5G/B5G scenarios. Our NDT, composed of a Graph Neural
Network (GNN), was evaluated across three different network topologies
consisting of 8, 16, and 30 nodes. It achieved lower MAPE values for URLLC and
eMBB slices, comparing latency predictions with actual latency after the
solution implementation. These values indicate high accuracy, demonstrating the
solution's effectiveness in generating precise insights into network
performance if a particular solution were implemented.

### 2. [Cross-Problem Solving for Network Optimization: Is Problem-Aware Learning the Key?](http://arxiv.org/pdf/2505.05067v1)

Authors: Ruihuai Liang, Bo Yang, Pengyu Chen, Xuelin Cao, Zhiwen Yu, H. Vincent Poor, Chau Yuen

As intelligent network services continue to diversify, ensuring efficient and
adaptive resource allocation in edge networks has become increasingly critical.
Yet the wide functional variations across services often give rise to new and
unforeseen optimization problems, rendering traditional manual modeling and
solver design both time-consuming and inflexible. This limitation reveals a key
gap between current methods and human solving - the inability to recognize and
understand problem characteristics. It raises the question of whether
problem-aware learning can bridge this gap and support effective cross-problem
generalization. To answer this question, we propose a problem-aware diffusion
(PAD) model, which leverages a problem-aware learning framework to enable
cross-problem generalization. By explicitly encoding the mathematical
formulations of optimization problems into token-level embeddings, PAD empowers
the model to understand and adapt to problem structures. Extensive experiments
across six diverse network optimization problems show that PAD generalizes well
to unseen problems while significantly improving solution quality and
feasibility. Meanwhile, an auxiliary constraint-aware module is designed to
enforce solution validity further. The experiments reveal that problem-aware
learning is promising for building general-purpose solvers for intelligent
network operation and resource management. Our code is open source at
https://github.com/qiyu3816/PAD.

### 3. [Temporal Spectrum Analysis for Multi-Constellation Space Domain Awareness](http://arxiv.org/pdf/2505.05149v1)

Authors: Mansour Naslcheraghi, Gunes Karabulut-Kurt

Space Domain Awareness (SDA) system has different major aspects including
continues and robust awareness from the network that is crucial for an
efficient control over all actors in space. The observability of the space
assets on the other hand requires efficient analysis on when and how observed
space objects can be controlled. This becomes crucial when real-world spatial
dynamics are taken into account as it introduces complexities into the system.
The real-world dynamics can reveal the structure of the network including
isolated and dominant stations. We propose a Temporal Spectrum Analysis (TSA)
scheme that takes into account a set of real-world parameters including actual
dynamics of the objects in space to analyze the structure of a ground-space
network that inherits temporal spectrum as the key element of design. We study
the potential interactions between multiple constellations using TSA and
conduct a comprehensive real-world simulations to quantify the structure of the
network. Numerical results show how the temporal spectrum of each satellite
affects the intra- and inter-constellation network structure including
interactions between ground stations and constellations.

### 4. [In-Situ Model Validation for Continuous Processes Using In-Network Computing](http://arxiv.org/pdf/2505.05184v1)

Authors: Ike Kunze, Dominik Scheurenberg, Liam Tirpitz, Sandra Geisler, Klaus Wehrle

The advancing industrial digitalization enables evolved process control
schemes that rely on accurate models learned through data-driven approaches.
While they provide high control performance and are robust to smaller
deviations, a larger change in process behavior can pose significant
challenges, in the worst case even leading to a damaged process plant. Hence,
it is important to frequently assess the fit between the model and the actual
process behavior. As the number of controlled processes and associated data
volumes increase, the need for lightweight and fast reacting assessment
solutions also increases. In this paper, we propose CIVIC, an in-network
computing-based solution for Continuous In-situ Validation of Industrial
Control models. In short, CIVIC monitors relevant process variables and detects
different process states through comparison with a priori knowledge about the
desired process behavior. This detection can then be leveraged to, e.g., shut
down the process or trigger a reconfiguration. We prototype CIVIC on an Intel
Tofino-based switch and apply it to a lab-scale water treatment plant. Our
results show that we can achieve a high detection accuracy, proving that such
monitoring systems are feasible and sensible.

### 5. [SDR-RDMA: Software-Defined Reliability Architecture for Planetary Scale RDMA Communication](http://arxiv.org/pdf/2505.05366v1)

Authors: Mikhail Khalilov, Siyuan Shen, Marcin Chrapek, Tiancheng Chen, Kenji Nakano, Peter-Jan Gootzen, Salvatore Di Girolamo, Rami Nudelman, Gil Bloch, Sreevatsa Anantharamu, Mahmoud Elhaddad, Jithin Jose, Abdul Kabbani, Scott Moe, Konstantin Taranov, Zhuolong Yu, Jie Zhang, Nicola Mazzoletti, Torsten Hoefler

RDMA is vital for efficient distributed training across datacenters, but
millisecond-scale latencies complicate the design of its reliability layer. We
show that depending on long-haul link characteristics, such as drop rate,
distance and bandwidth, the widely used Selective Repeat algorithm can be
inefficient, warranting alternatives like Erasure Coding. To enable such
alternatives on existing hardware, we propose SDR-RDMA, a software-defined
reliability stack for RDMA. Its core is a lightweight SDR SDK that extends
standard point-to-point RDMA semantics -- fundamental to AI networking stacks
-- with a receive buffer bitmap. SDR bitmap enables partial message completion
to let applications implement custom reliability schemes tailored to specific
deployments, while preserving zero-copy RDMA benefits. By offloading the SDR
backend to NVIDIA's Data Path Accelerator (DPA), we achieve line-rate
performance, enabling efficient inter-datacenter communication and advancing
reliability innovation for intra-datacenter training.

### 6. [A Weighted Byzantine Fault Tolerance Consensus Driven Trusted Multiple Large Language Models Network](http://arxiv.org/pdf/2505.05103v1)

Authors: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dongcheng Zhao, Dusit Niyato, Hongfang Yu, Schahram Dustdar

Large Language Models (LLMs) have achieved remarkable success across a wide
range of applications. However, individual LLMs often produce inconsistent,
biased, or hallucinated outputs due to limitations in their training corpora
and model architectures. Recently, collaborative frameworks such as the
Multi-LLM Network (MultiLLMN) have been introduced, enabling multiple LLMs to
interact and jointly respond to user queries. Nevertheless, MultiLLMN
architectures raise critical concerns regarding the reliability and security of
the generated content, particularly in open environments where malicious or
compromised LLMs may be present. Moreover, reliance on centralized coordination
undermines system efficiency and introduces single points of failure. In this
paper, we propose a novel Trusted MultiLLMN framework, driven by a Weighted
Byzantine Fault Tolerance (WBFT) blockchain consensus mechanism, to ensure the
reliability, security, and efficiency of multi-LLM collaboration. In WBFT,
voting weights are adaptively assigned to each LLM based on its response
quality and trustworthiness, incentivizing reliable behavior, and reducing the
impact of malicious nodes. Extensive simulations demonstrate that WBFT
significantly improves both consensus security and efficiency compared to
classical and modern consensus mechanisms, particularly under wireless network
conditions. Furthermore, our evaluations reveal that Trusted MultiLLMN
supported by WBFT can deliver higher-quality and more credible responses than
both single LLMs and conventional MultiLLMNs, thereby providing a promising
path toward building robust, decentralized AI collaboration networks.

### 7. [AI and Vision based Autonomous Navigation of Nano-Drones in Partially-Known Environments](http://arxiv.org/pdf/2505.04972v1)

Authors: Mattia Sartori, Chetna Singhal, Neelabhro Roy, Davide Brunelli, James Gross

The miniaturisation of sensors and processors, the advancements in connected
edge intelligence, and the exponential interest in Artificial Intelligence are
boosting the affirmation of autonomous nano-size drones in the Internet of
Robotic Things ecosystem. However, achieving safe autonomous navigation and
high-level tasks such as exploration and surveillance with these tiny platforms
is extremely challenging due to their limited resources. This work focuses on
enabling the safe and autonomous flight of a pocket-size, 30-gram platform
called Crazyflie 2.1 in a partially known environment. We propose a novel
AI-aided, vision-based reactive planning method for obstacle avoidance under
the ambit of Integrated Sensing, Computing and Communication paradigm. We deal
with the constraints of the nano-drone by splitting the navigation task into
two parts: a deep learning-based object detector runs on the edge (external
hardware) while the planning algorithm is executed onboard. The results show
the ability to command the drone at $\sim8$ frames-per-second and a model
performance reaching a COCO mean-average-precision of $60.8$. Field experiments
demonstrate the feasibility of the solution with the drone flying at a top
speed of $1$ m/s while steering away from an obstacle placed in an unknown
position and reaching the target destination. The outcome highlights the
compatibility of the communication delay and the model performance with the
requirements of the real-time navigation task. We provide a feasible
alternative to a fully onboard implementation that can be extended to
autonomous exploration with nano-drones.

### Robotics

### 1. [SatAOI: Delimitating Area of Interest for Swing-Arm Troweling Robot for Construction](http://arxiv.org/pdf/2505.04871v1)

Authors: Jia-Rui Lin, Shaojie Zhou, Peng Pan, Ruijia Cai, Gang Chen

In concrete troweling for building construction, robots can significantly
reduce workload and improve automation level. However, as a primary task of
coverage path planning (CPP) for troweling, delimitating area of interest (AOI)
in complex scenes is still challenging, especially for swing-arm robots with
more complex working modes. Thus, this research proposes an algorithm to
delimitate AOI for swing-arm troweling robot (SatAOI algorithm). By analyzing
characteristics of the robot and obstacle maps, mathematical models and
collision principles are established. On this basis, SatAOI algorithm achieves
AOI delimitation by global search and collision detection. Experiments on
different obstacle maps indicate that AOI can be effectively delimitated in
scenes under different complexity, and the algorithm can fully consider the
connectivity of obstacle maps. This research serves as a foundation for CPP
algorithm and full process simulation of swing-arm troweling robots.

### 2. [Real-Time Model Predictive Control of Vehicles with Convex-Polygon-Aware Collision Avoidance in Tight Spaces](http://arxiv.org/pdf/2505.04935v1)

Authors: Haruki Kojima, Kohei Honda, Hiroyuki Okuda, Tatsuya Suzuki

This paper proposes vehicle motion planning methods with obstacle avoidance
in tight spaces by incorporating polygonal approximations of both the vehicle
and obstacles into a model predictive control (MPC) framework. Representing
these shapes is crucial for navigation in tight spaces to ensure accurate
collision detection. However, incorporating polygonal approximations leads to
disjunctive OR constraints in the MPC formulation, which require a mixed
integer programming and cause significant computational cost. To overcome this,
we propose two different collision-avoidance constraints that reformulate the
disjunctive OR constraints as tractable conjunctive AND constraints: (1) a
Support Vector Machine (SVM)-based formulation that recasts collision avoidance
as a SVM optimization problem, and (2) a Minimum Signed Distance to Edges
(MSDE) formulation that leverages minimum signed-distance metrics. We validate
both methods through extensive simulations, including tight-space parking
scenarios and varied-shape obstacle courses, as well as hardware experiments on
an RC-car platform. Our results demonstrate that the SVM-based approach
achieves superior navigation accuracy in constrained environments; the MSDE
approach, by contrast, runs in real time with only a modest reduction in
collision-avoidance performance.

### 3. [Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking](http://arxiv.org/pdf/2505.04978v1)

Authors: Yongpeng Jiang, Mingrui Yu, Xinghao Zhu, Masayoshi Tomizuka, Xiang Li

Robotic dexterous in-hand manipulation, where multiple fingers dynamically
make and break contact, represents a step toward human-like dexterity in
real-world robotic applications. Unlike learning-based approaches that rely on
large-scale training or extensive data collection for each specific task,
model-based methods offer an efficient alternative. Their online computing
nature allows for ready application to new tasks without extensive retraining.
However, due to the complexity of physical contacts, existing model-based
methods encounter challenges in efficient online planning and handling modeling
errors, which limit their practical applications. To advance the effectiveness
and robustness of model-based contact-rich in-hand manipulation, this paper
proposes a novel integrated framework that mitigates these limitations. The
integration involves two key aspects: 1) integrated real-time planning and
tracking achieved by a hierarchical structure; and 2) joint optimization of
motions and contacts achieved by integrated motion-contact modeling.
Specifically, at the high level, finger motion and contact force references are
jointly generated using contact-implicit model predictive control. The
high-level module facilitates real-time planning and disturbance recovery. At
the low level, these integrated references are concurrently tracked using a
hand force-motion model and actual tactile feedback. The low-level module
compensates for modeling errors and enhances the robustness of manipulation.
Extensive experiments demonstrate that our approach outperforms existing
model-based methods in terms of accuracy, robustness, and real-time
performance. Our method successfully completes five challenging tasks in
real-world environments, even under appreciable external disturbances.

### 4. [CPP-DIP: Multi-objective Coverage Path Planning for MAVs in Dispersed and Irregular Plantations](http://arxiv.org/pdf/2505.04989v1)

Authors: Weijie Kuang, Hann Woei Ho, Ye Zhou

Coverage Path Planning (CPP) is vital in precision agriculture to improve
efficiency and resource utilization. In irregular and dispersed plantations,
traditional grid-based CPP often causes redundant coverage over non-vegetated
areas, leading to waste and pollution. To overcome these limitations, we
propose CPP-DIP, a multi-objective CPP framework designed for Micro Air
Vehicles (MAVs). The framework transforms the CPP task into a Traveling
Salesman Problem (TSP) and optimizes flight paths by minimizing travel
distance, turning angles, and intersection counts. Unlike conventional
approaches, our method does not rely on GPS-based environmental modeling.
Instead, it uses aerial imagery and a Histogram of Oriented Gradients
(HOG)-based approach to detect trees and extract image coordinates. A
density-aware waypoint strategy is applied: Kernel Density Estimation (KDE) is
used to reduce redundant waypoints in dense regions, while a greedy algorithm
ensures complete coverage in sparse areas. To verify the generality of the
framework, we solve the resulting TSP using three different methods: Greedy
Heuristic Insertion (GHI), Ant Colony Optimization (ACO), and Monte Carlo
Reinforcement Learning (MCRL). Then an object-based optimization is applied to
further refine the resulting path. Additionally, CPP-DIP integrates ForaNav,
our insect-inspired navigation method, for accurate tree localization and
tracking. The experimental results show that MCRL offers a balanced solution,
reducing the travel distance by 16.9 % compared to ACO while maintaining a
similar performance to GHI. It also improves path smoothness by reducing
turning angles by 28.3 % and 59.9 % relative to ACO and GHI, respectively, and
effectively eliminates intersections. These results confirm the robustness and
effectiveness of CPP-DIP in different TSP solvers.

### 5. [Online Velocity Profile Generation and Tracking for Sampling-Based Local Planning Algorithms in Autonomous Racing Environments](http://arxiv.org/pdf/2505.05157v1)

Authors: Alexander Langmann, Levent Ögretmen, Frederik Werner, Johannes Betz

This work presents an online velocity planner for autonomous racing that
adapts to changing dynamic constraints, such as grip variations from tire
temperature changes and rubber accumulation. The method combines a
forward-backward solver for online velocity optimization with a novel spatial
sampling strategy for local trajectory planning, utilizing a three-dimensional
track representation. The computed velocity profile serves as a reference for
the local planner, ensuring adaptability to environmental and vehicle dynamics.
We demonstrate the approach's robust performance and computational efficiency
in racing scenarios and discuss its limitations, including sensitivity to
deviations from the predefined racing line and high jerk characteristics of the
velocity profile.

### 6. [CottonSim: Development of an autonomous visual-guided robotic cotton-picking system in the Gazebo](http://arxiv.org/pdf/2505.05317v1)

Authors: Thevathayarajh Thayananthan, Xin Zhang, Yanbo Huang, Jingdao Chen, Nuwan K. Wijewardane, Vitor S. Martins, Gary D. Chesser, Christopher T. Goodin

In this study, an autonomous visual-guided robotic cotton-picking system,
built on a Clearpath's Husky robot platform and the Cotton-Eye perception
system, was developed in the Gazebo robotic simulator. Furthermore, a virtual
cotton farm was designed and developed as a Robot Operating System (ROS 1)
package to deploy the robotic cotton picker in the Gazebo environment for
simulating autonomous field navigation. The navigation was assisted by the map
coordinates and an RGB-depth camera, while the ROS navigation algorithm
utilized a trained YOLOv8n-seg model for instance segmentation. The model
achieved a desired mean Average Precision (mAP) of 85.2%, a recall of 88.9%,
and a precision of 93.0% for scene segmentation. The developed ROS navigation
packages enabled our robotic cotton-picking system to autonomously navigate
through the cotton field using map-based and GPS-based approaches, visually
aided by a deep learning-based perception system. The GPS-based navigation
approach achieved a 100% completion rate (CR) with a threshold of 5 x 10^-6
degrees, while the map-based navigation approach attained a 96.7% CR with a
threshold of 0.25 m. This study establishes a fundamental baseline of
simulation for future agricultural robotics and autonomous vehicles in cotton
farming and beyond. CottonSim code and data are released to the research
community via GitHub: https://github.com/imtheva/CottonSim

### 7. [DSDrive: Distilling Large Language Model for Lightweight End-to-End Autonomous Driving with Unified Reasoning and Planning](http://arxiv.org/pdf/2505.05360v1)

Authors: Wenru Liu, Pei Liu, Jun Ma

We present DSDrive, a streamlined end-to-end paradigm tailored for
integrating the reasoning and planning of autonomous vehicles into a unified
framework. DSDrive leverages a compact LLM that employs a distillation method
to preserve the enhanced reasoning capabilities of a larger-sized vision
language model (VLM). To effectively align the reasoning and planning tasks, a
waypoint-driven dual-head coordination module is further developed, which
synchronizes dataset structures, optimization objectives, and the learning
process. By integrating these tasks into a unified framework, DSDrive anchors
on the planning results while incorporating detailed reasoning insights,
thereby enhancing the interpretability and reliability of the end-to-end
pipeline. DSDrive has been thoroughly tested in closed-loop simulations, where
it performs on par with benchmark models and even outperforms in many key
metrics, all while being more compact in size. Additionally, the computational
efficiency of DSDrive (as reflected in its time and memory requirements during
inference) has been significantly enhanced. Evidently thus, this work brings
promising aspects and underscores the potential of lightweight systems in
delivering interpretable and efficient solutions for AD.

### 8. [CubeDAgger: Improved Robustness of Interactive Imitation Learning without Violation of Dynamic Stability](http://arxiv.org/pdf/2505.04897v1)

Authors: Taisuke Kobayashi

Interactive imitation learning makes an agent's control policy robust by
stepwise supervisions from an expert. The recent algorithms mostly employ
expert-agent switching systems to reduce the expert's burden by limitedly
selecting the supervision timing. However, the precise selection is difficult
and such a switching causes abrupt changes in actions, damaging the dynamic
stability. This paper therefore proposes a novel method, so-called CubeDAgger,
which improves robustness while reducing dynamic stability violations by making
three improvements to a baseline method, EnsembleDAgger. The first improvement
adds a regularization to explicitly activate the threshold for deciding the
supervision timing. The second transforms the expert-agent switching system to
an optimal consensus system of multiple action candidates. Third,
autoregressive colored noise to the actions is introduced to make the
stochastic exploration consistent over time. These improvements are verified by
simulations, showing that the learned policies are sufficiently robust while
maintaining dynamic stability during interaction.

### 9. [An Efficient Method for Accurate Pose Estimation and Error Correction of Cuboidal Objects](http://arxiv.org/pdf/2505.04962v1)

Authors: Utsav Rai, Hardik Mehta, Vismay Vakharia, Aditya Choudhary, Amit Parmar, Rolif Lima, Kaushik Das

The proposed system outlined in this paper is a solution to a use case that
requires the autonomous picking of cuboidal objects from an organized or
unorganized pile with high precision. This paper presents an efficient method
for precise pose estimation of cuboid-shaped objects, which aims to reduce
errors in target pose in a time-efficient manner. Typical pose estimation
methods like global point cloud registrations are prone to minor pose errors
for which local registration algorithms are generally used to improve pose
accuracy. However, due to the execution time overhead and uncertainty in the
error of the final achieved pose, an alternate, linear time approach is
proposed for pose error estimation and correction. This paper presents an
overview of the solution followed by a detailed description of individual
modules of the proposed algorithm.

### 10. [CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations](http://arxiv.org/pdf/2505.04999v1)

Authors: Anthony Liang, Pavel Czempin, Matthew Hong, Yutai Zhou, Erdem Biyik, Stephen Tu

Learning robot policies using imitation learning requires collecting large
amounts of costly action-labeled expert demonstrations, which fundamentally
limits the scale of training data. A promising approach to address this
bottleneck is to harness the abundance of unlabeled observations-e.g., from
video demonstrations-to learn latent action labels in an unsupervised way.
However, we find that existing methods struggle when applied to complex robot
tasks requiring fine-grained motions. We design continuous latent action models
(CLAM) which incorporate two key ingredients we find necessary for learning to
solve complex continuous control tasks from unlabeled observation data: (a)
using continuous latent action labels instead of discrete representations, and
(b) jointly training an action decoder to ensure that the latent action space
can be easily grounded to real actions with relatively few labeled examples.
Importantly, the labeled examples can be collected from non-optimal play data,
enabling CLAM to learn performant policies without access to any action-labeled
expert data. We demonstrate on continuous control benchmarks in DMControl
(locomotion) and MetaWorld (manipulation), as well as on a real WidowX robot
arm that CLAM significantly outperforms prior state-of-the-art methods,
remarkably with a 2-3x improvement in task success rate compared to the best
baseline. Videos and code can be found at clamrobot.github.io.

### Software Engineering

### 1. [Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware](http://arxiv.org/pdf/2505.05057v1)

Authors: Yujia Chen, Mingyu Chen, Cuiyun Gao, Zhihan Jiang, Zhongqi Li, Yuchi Ma

Application Programming Interfaces (APIs) are crucial in modern software
development. Large Language Models (LLMs) assist in automated code generation
but often struggle with API hallucination, including invoking non-existent APIs
and misusing existing ones in practical development scenarios. Existing studies
resort to Retrieval-Augmented Generation (RAG) methods for mitigating the
hallucination issue, but tend to fail since they generally ignore the
structural dependencies in practical projects and do not indeed validate
whether the generated APIs are available or not. To address these limitations,
we propose MARIN, a framework for mitigating API hallucination in code
generated by LLMs with hierarchical dependency aware. MARIN consists of two
phases: Hierarchical Dependency Mining, which analyzes local and global
dependencies of the current function, aiming to supplement comprehensive
project context in LLMs input, and Dependency Constrained Decoding, which
utilizes mined dependencies to adaptively constrain the generation process,
aiming to ensure the generated APIs align with the projects specifications. To
facilitate the evaluation of the degree of API hallucination, we introduce a
new benchmark APIHulBench and two new metrics including Micro Hallucination
Number (MiHN) and Macro Hallucination Rate (MaHR). Experiments on six
state-of-the-art LLMs demonstrate that MARIN effectively reduces API
hallucinations, achieving an average decrease of 67.52% in MiHN and 73.56% in
MaHR compared to the RAG approach. Applied to Huaweis internal projects and two
proprietary LLMs, MARIN achieves average decreases of 57.33% in MiHN and 59.41%
in MaHR.

### 2. [Overcoming the hurdle of legal expertise: A reusable model for smartwatch privacy policies](http://arxiv.org/pdf/2505.05214v1)

Authors: Constantin Buschhaus, Arvid Butting, Judith Michael, Verena Nitsch, Sebastian Pütz, Bernhard Rumpe, Carolin Stellmacher, Sabine Theis

Regulations for privacy protection aim to protect individuals from the
unauthorized storage, processing, and transfer of their personal data but
oftentimes fail in providing helpful support for understanding these
regulations. To better communicate privacy policies for smartwatches, we need
an in-depth understanding of their concepts and provide better ways to enable
developers to integrate them when engineering systems. Up to now, no conceptual
model exists covering privacy statements from different smartwatch
manufacturers that is reusable for developers. This paper introduces such a
conceptual model for privacy policies of smartwatches and shows its use in a
model-driven software engineering approach to create a platform for data
visualization of wearable privacy policies from different smartwatch
manufacturers. We have analyzed the privacy policies of various manufacturers
and extracted the relevant concepts. Moreover, we have checked the model with
lawyers for its correctness, instantiated it with concrete data, and used it in
a model-driven software engineering approach to create a platform for data
visualization. This reusable privacy policy model can enable developers to
easily represent privacy policies in their systems. This provides a foundation
for more structured and understandable privacy policies which, in the long run,
can increase the data sovereignty of application users.

### 3. [TS-Detector : Detecting Feature Toggle Usage Patterns](http://arxiv.org/pdf/2505.05326v1)

Authors: Tajmilur Rahman, Mengzhe Fei, Tushar Sharma, Chanchal Roy

Feature toggles enable developers to control feature states, allowing the
features to be released to a limited group of users while preserving overall
software functionality. The absence of comprehensive best practices for feature
toggle usage often results in improper implementation, causing code quality
issues. Although certain feature toggle usage patterns are prone to toggle
smells, there is no tool as of today for software engineers to detect toggle
usage patterns from the source code. This paper presents a tool TS-Detector to
detect five different toggle usage patterns across ten open-source software
projects in six different programming languages. We conducted a manual
evaluation and results show that the true positive rates of detecting Spread,
Nested, and Dead toggles are 80%, 86.4%, and 66.6% respectively, and the true
negative rate of Mixed and Enum usages was 100%. The tool can be downloaded
from its GitHub repository and can be used following the instructions provided
there.

### 4. [Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents](http://arxiv.org/pdf/2505.05283v1)

Authors: Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Bin Shi

Code large language models (CodeLLMs) and agents have shown great promise in
tackling complex software engineering tasks.Compared to traditional software
engineering methods, CodeLLMs and agents offer stronger abilities, and can
flexibly process inputs and outputs in both natural and code. Benchmarking
plays a crucial role in evaluating the capabilities of CodeLLMs and agents,
guiding their development and deployment. However, despite their growing
significance, there remains a lack of comprehensive reviews of benchmarks for
CodeLLMs and agents. To bridge this gap, this paper provides a comprehensive
review of existing benchmarks for CodeLLMs and agents, studying and analyzing
181 benchmarks from 461 relevant papers, covering the different phases of the
software development life cycle (SDLC). Our findings reveal a notable imbalance
in the coverage of current benchmarks, with approximately 60% focused on the
software development phase in SDLC, while requirements engineering and software
design phases receive minimal attention at only 5% and 3%, respectively.
Additionally, Python emerges as the dominant programming language across the
reviewed benchmarks. Finally, this paper highlights the challenges of current
research and proposes future directions, aiming to narrow the gap between the
theoretical capabilities of CodeLLMs and agents and their application in
real-world scenarios.

### Social and Information Networks

### 1. [Community and hyperedge inference in multiple hypergraphs](http://arxiv.org/pdf/2505.04967v1)

Authors: Li Ni, Ziqi Deng, Lin Mu, Lei Zhang, Wenjian Luo, Yiwen Zhang

Hypergraphs, capable of representing high-order interactions via hyperedges,
have become a powerful tool for modeling real-world biological and social
systems. Inherent relationships within these real-world systems, such as the
encoding relationship between genes and their protein products, drive the
establishment of interconnections between multiple hypergraphs. Here, we
demonstrate how to utilize those interconnections between multiple hypergraphs
to synthesize integrated information from multiple higher-order systems,
thereby enhancing understanding of underlying structures. We propose a model
based on the stochastic block model, which integrates information from multiple
hypergraphs to reveal latent high-order structures. Real-world hyperedges
exhibit preferential attachment, where certain nodes dominate hyperedge
formation. To characterize this phenomenon, our model introduces hyperedge
internal degree to quantify nodes' contributions to hyperedge formation. This
model is capable of mining communities, predicting missing hyperedges of
arbitrary sizes within hypergraphs, and inferring inter-hypergraph edges
between hypergraphs. We apply our model to high-order datasets to evaluate its
performance. Experimental results demonstrate strong performance of our model
in community detection, hyperedge prediction, and inter-hypergraph edge
prediction tasks. Moreover, we show that our model enables analysis of multiple
hypergraphs of different types and supports the analysis of a single hypergraph
in the absence of inter-hypergraph edges. Our work provides a practical and
flexible tool for analyzing multiple hypergraphs, greatly advancing the
understanding of the organization in real-world high-order systems.

### 2. [UKElectionNarratives: A Dataset of Misleading Narratives Surrounding Recent UK General Elections](http://arxiv.org/pdf/2505.05459v1)

Authors: Fatima Haouari, Carolina Scarton, Nicolò Faggiani, Nikolaos Nikolaidis, Bonka Kotseva, Ibrahim Abu Farha, Jens Linge, Kalina Bontcheva

Misleading narratives play a crucial role in shaping public opinion during
elections, as they can influence how voters perceive candidates and political
parties. This entails the need to detect these narratives accurately. To
address this, we introduce the first taxonomy of common misleading narratives
that circulated during recent elections in Europe. Based on this taxonomy, we
construct and analyse UKElectionNarratives: the first dataset of
human-annotated misleading narratives which circulated during the UK General
Elections in 2019 and 2024. We also benchmark Pre-trained and Large Language
Models (focusing on GPT-4o), studying their effectiveness in detecting
election-related misleading narratives. Finally, we discuss potential use cases
and make recommendations for future research directions using the proposed
codebook and dataset.

### Systems and Control

### 1. [Learning Economic Model Predictive Control via Clustering and Kernel-Based Lipschitz Regression](http://arxiv.org/pdf/2505.04904v1)

Authors: Weiliang Xiong, Defeng He, Haiping Du

This paper presents a novel learning economic model predictive control scheme
for uncertain nonlinear systems subject to input and state constraints and
unknown dynamics. We design a fast and accurate Lipschitz regression method
using input and output data that combines clustering and kernel regression to
learn the unknown dynamics. In each cluster, the parallel convex optimization
problems are solved to estimate the kernel weights and reduce the Lipschitz
constant of the predictor, hence limiting the error propagation in the
prediction horizon. We derive the two different bounds of learning errors in
deterministic and probabilistic forms and customize a new robust
constraint-tightening strategy for the discontinuous predictor. Then, the
learning economic model predictive control algorithm is formulated by
introducing a stabilized optimization problem to construct a Lyapunov function.
Sufficient conditions are derived to ensure the recursive feasibility and
input-to-state stability of the closed-loop system. The effectiveness of the
proposed algorithm is verified by simulations of a numerical example and a
continuously stirred tank reactor.

### 2. [Enhanced Robust Tracking Control: An Online Learning Approach](http://arxiv.org/pdf/2505.05036v1)

Authors: Ao Jin, Weijian Zhao, Yifeng Ma, Panfeng Huang, Fan Zhang

This work focuses the tracking control problem for nonlinear systems
subjected to unknown external disturbances. Inspired by contraction theory, a
neural network-dirven CCM synthesis is adopted to obtain a feedback controller
that could track any feasible trajectory. Based on the observation that the
system states under continuous control input inherently contain embedded
information about unknown external disturbances, we propose an online learning
scheme that captures the disturbances dyanmics from online historical data and
embeds the compensation within the CCM controller. The proposed scheme operates
as a plug-and-play module that intrinsically enhances the tracking performance
of CCM synthesis. The numerical simulations on tethered space robot and PVTOL
demonstrate the effectiveness of proposed scheme. The source code of the
proposed online learning scheme can be found at
https://github.com/NPU-RCIR/Online_CCM.git.

### 3. [Predictive Control of EV Overnight Charging with Multi-Session Flexibility](http://arxiv.org/pdf/2505.05087v1)

Authors: Felix Wieberneit, Emanuele Crisostomi, Anthony Quinn, Robert Shorten

The majority of electric vehicles (EVs) are charged domestically overnight,
where the precise timing of power allocation is not important to the user, thus
representing a source of flexibility that can be leveraged by charging control
algorithms. In this paper, we relax the common assumption, that EVs require
full charge every morning, enabling additional flexibility to defer charging of
surplus energy to subsequent nights, which can enhance the performance of
controlled charging. In particular, we consider a simple domestic smart plug,
scheduling power delivery with the objective to minimize CO$_2$ emissions over
prediction horizons of multiple sessions -- up to seven days ahead -- utilising
model predictive control (MPC). Based on carbon intensity data from the UK
National Grid, we demonstrate significant potential for emission reductions
with multi-session planning of 40 to 46\% compared to uncontrolled charging and
19 to 26\% compared to single-session planning. Furthermore, we assess, how the
driving and charging behaviour of EV users affects the available flexibility
and consequentially the potential for emission reductions. Finally, using grid
carbon intensity data from 14 different UK regions, we report significant
variations in absolute emission reductions based on the local energy mix.

### 4. [Day-Ahead Bidding Strategies for Wind Farm Operators under a One-Price Balancing Scheme](http://arxiv.org/pdf/2505.05153v1)

Authors: Max Bruninx, Timothy Verstraeten, Jalal Kazempour, Jan Helsen

We study day-ahead bidding strategies for wind farm operators under a
one-price balancing scheme, prevalent in European electricity markets. In this
setting, the profit-maximising strategy becomes an all-or-nothing strategy,
aiming to take advantage of open positions in the balancing market. However,
balancing prices are difficult, if not impossible, to forecast in the day-ahead
stage and large open positions can affect the balancing price by changing the
direction of the system imbalance. This paper addresses day-ahead bidding as a
decision-making problem under uncertainty, with the objective of maximising the
expected profit while reducing the imbalance risk related to the strategy. To
this end, we develop a stochastic optimisation problem with explicit
constraints on the positions in the balancing market, providing risk
certificates, and derive an analytical solution to this problem. Moreover, we
show how the price-impact of the trading strategy on the balancing market can
be included in the ex-post evaluation. Using real data from the Belgian
electricity market and an offshore wind farm in the North Sea, we demonstrate
that the all-or-nothing strategy negatively impacts the balancing price,
resulting in long-term losses for the wind farm. Our risk-constrained strategy,
however, can still significantly enhance operational profit compared to
traditional point-forecast bidding.

### 5. [Adaptive Biased User Scheduling for Heterogeneous Wireless Federate Learning Network](http://arxiv.org/pdf/2505.05231v1)

Authors: Changxiang Wu, Yijing Ren, Daniel K. C. So, Jie Tang

Federated Learning (FL) has revolutionized collaborative model training in
distributed networks, prioritizing data privacy and communication efficiency.
This paper investigates efficient deployment of FL in wireless heterogeneous
networks, focusing on strategies to accelerate convergence despite stragglers.
The primary objective is to minimize long-term convergence wall-clock time
through optimized user scheduling and resource allocation. While stragglers may
introduce delays in a single round, their inclusion can expedite subsequent
rounds, particularly when they possess critical information. Moreover,
balancing single-round duration with the number of cumulative rounds,
compounded by dynamic training and transmission conditions, necessitates a
novel approach beyond conventional optimization solutions. To tackle these
challenges, convergence analysis with respect to adaptive and biased scheduling
is derived. Then, by factoring in real-time system and statistical information,
including diverse energy constraints and users' energy harvesting capabilities,
a deep reinforcement learning approach, empowered by proximal policy
optimization, is employed to adaptively select user sets. For the scheduled
users, Lagrangian decomposition is applied to optimize local resource
utilization, further enhancing system efficiency. Simulation results validate
the effectiveness and robustness of the proposed framework for various FL
tasks, demonstrating reduced task time compared to existing benchmarks under
various settings.

### 6. [High Altitude Platform-Based Caching and Multicasting for Rural Connectivity](http://arxiv.org/pdf/2505.05251v1)

Authors: Yongqiang Zhang, Mustafa A. Kishk, Mohamed-Slim Alouini

Providing efficient and reliable content delivery in rural areas remains a
significant challenge due to the lack of communication infrastructure. To
bridge the digital divide, this paper investigates the potential of leveraging
multiple high-altitude platforms (HAPs) for energy-efficient content delivery
in wide rural regions. Each caching-enabled HAP is equipped with both
Free-Space Optical (FSO) transceivers for backhaul links and Radio Frequency
(RF) antenna arrays for access links. To further enhance network efficiency, we
consider a network coding-based multicasting scheme, where different types of
content are treated as distinct multicast sessions. With the objective of
minimizing long-term power cost, we propose a hierarchical framework that
integrates deep reinforcement learn-ing (DRL) and convex optimization to
jointly optimize dynamic caching strategies and resource allocation across the
network. Simulation results demonstrate that our approach significantly reduces
power cost compared to several baseline approaches, providing a practical
solution for improving rural connectivity.

### 7. [CV-MP: Max-Pressure Control in Heterogeneously Distributed and Partially Connected Vehicle Environments](http://arxiv.org/pdf/2505.05258v1)

Authors: Chaopeng Tan, Dingshan Sun, Hao Liu, Marco Rinaldi, Hans van Lint

Max-pressure (MP) control has emerged as a prominent real-time network
traffic signal control strategy due to its simplicity, decentralized structure,
and theoretical guarantees of network queue stability. Meanwhile, advances in
connected vehicle (CV) technology have sparked extensive research into CV-based
traffic signal control. Despite these developments, few studies have
investigated MP control in heterogeneously distributed and partially CV
environments while ensuring network queue stability. To address these research
gaps, we propose a CV-based MP control (CV-MP) method that leverages real-time
CV travel time information to compute the pressure, thereby incorporating both
the spatial distribution and temporal delays of vehicles, unlike existing
approaches that utilized only spatial distribution or temporal delays. In
particular, we establish sufficient conditions for road network queue stability
that are compatible with most existing MP control methods. Moreover, we
pioneered the proof of network queue stability even if the vehicles are only
partially connected and heterogeneously distributed, and gave a necessary
condition of CV observation for maintaining the stability. Evaluation results
on an Amsterdam corridor show that CV-MP significantly reduces vehicle delays
compared to both actuated control and conventional MP control across various CV
penetration rates. Moreover, in scenarios with dynamic traffic demand, CV-MP
achieves lower spillover peaks even with low and heterogeneous CV penetration
rates, further highlighting its effectiveness and robustness.

### 8. [Optimal Microgrid Sizing of Offshore Renewable Energy Sources for Offshore Platforms and Coastal Communities](http://arxiv.org/pdf/2505.05305v1)

Authors: Ann Mary Toms, Xingpeng Li, Kaushik Rajashekara

The global energy landscape is undergoing a transformative shift towards
renewable energy and advanced storage solutions, driven by the urgent need for
sustainable and resilient power systems. Isolated offshore communities, such as
islands and offshore platforms, which traditionally rely on mainland grids or
diesel generators, stand to gain significantly from renewable energy
integration. Promising offshore renewable technologies include wind turbines,
wave and tidal energy converters, and floating photovoltaic systems, paired
with a storage solution like battery energy storage systems. This paper
introduces a renewable energy microgrid optimizer (REMO), a tool designed to
identify the optimal sizes of renewable generation and storage resources for
offshore microgrids. A key challenge in such models is accurately accounting
for battery degradation costs. To address this, the REMO model integrates a
deep neural network-based battery degradation (DNN-BD) module, which factors in
variables like ambient temperature, charge/discharge rates, state of charge,
depth of discharge and battery health. Simulations on six test regions
demonstrate that the REMO-DNN-BD approach minimizes lifetime energy costs while
maintaining high reliability and sustainability, making it a viable design
solution for offshore microgrid systems.

### 9. [Approximation-free Control for Signal Temporal Logic Specifications using Spatiotemporal Tubes](http://arxiv.org/pdf/2505.05323v1)

Authors: Ratnangshu Das, Subhodeep Choudhury, Pushpak Jagtap

This paper presents a spatiotemporal tube (STT)-based control framework for
satisfying Signal Temporal Logic (STL) specifications in unknown control-affine
systems. We formulate STL constraints as a robust optimization problem (ROP)
and recast it as a scenario optimization program (SOP) to construct STTs with
formal correctness guarantees. We also propose a closed-form control law that
operates independently of the system dynamics, and ensures the system
trajectory evolves within the STTs, thereby satisfying the STL specifications.
The proposed approach is validated through case studies and comparisons with
state-of-the-art methods, demonstrating superior computational efficiency,
trajectory quality, and applicability to complex STL tasks.

### 10. [Learning Linearized Models from Nonlinear Systems under Initialization Constraints with Finite Data](http://arxiv.org/pdf/2505.04954v1)

Authors: Lei Xin, Baike She, Qi Dou, George Chiu, Shreyas Sundaram

The identification of a linear system model from data has wide applications
in control theory. The existing work that provides finite sample guarantees for
linear system identification typically uses data from a single long system
trajectory under i.i.d. random inputs, and assumes that the underlying dynamics
is truly linear. In contrast, we consider the problem of identifying a
linearized model when the true underlying dynamics is nonlinear, given that
there is a certain constraint on the region where one can initialize the
experiments. We provide a multiple trajectories-based deterministic data
acquisition algorithm followed by a regularized least squares algorithm, and
provide a finite sample error bound on the learned linearized dynamics. Our
error bound shows that one can consistently learn the linearized dynamics, and
demonstrates a trade-off between the error due to nonlinearity and the error
due to noise. We validate our results through numerical experiments, where we
also show the potential insufficiency of linear system identification using a
single trajectory with i.i.d. random inputs, when nonlinearity does exist.

### Machine Learning (Statistics Category)

### 1. [Generalization Analysis for Contrastive Representation Learning under Non-IID Settings](http://arxiv.org/pdf/2505.04937v1)

Authors: Nong Minh Hieu, Antoine Ledent

Contrastive Representation Learning (CRL) has achieved impressive success in
various domains in recent years. Nevertheless, the theoretical understanding of
the generalization behavior of CRL is limited. Moreover, to the best of our
knowledge, the current literature only analyzes generalization bounds under the
assumption that the data tuples used for contrastive learning are independently
and identically distributed. However, in practice, we are often limited to a
fixed pool of reusable labeled data points, making it inevitable to recycle
data across tuples to create sufficiently large datasets. Therefore, the
tuple-wise independence condition imposed by previous works is invalidated. In
this paper, we provide a generalization analysis for the CRL framework under
non-$i.i.d.$ settings that adheres to practice more realistically. Drawing
inspiration from the literature on U-statistics, we derive generalization
bounds which indicate the required number of samples in each class scales as
the logarithm of the covering number of the class of learnable feature
representations associated to each class. Next, we apply our main results to
derive excess risk bounds for common function classes such as linear maps and
neural networks.

### 2. [Conformal Prediction with Cellwise Outliers: A Detect-then-Impute Approach](http://arxiv.org/pdf/2505.04986v1)

Authors: Qian Peng, Yajie Bao, Haojie Ren, Zhaojun Wang, Changliang Zou

Conformal prediction is a powerful tool for constructing prediction intervals
for black-box models, providing a finite sample coverage guarantee for
exchangeable data. However, this exchangeability is compromised when some
entries of the test feature are contaminated, such as in the case of cellwise
outliers. To address this issue, this paper introduces a novel framework called
detect-then-impute conformal prediction. This framework first employs an
outlier detection procedure on the test feature and then utilizes an imputation
method to fill in those cells identified as outliers. To quantify the
uncertainty in the processed test feature, we adaptively apply the detection
and imputation procedures to the calibration set, thereby constructing
exchangeable features for the conformal prediction interval of the test label.
We develop two practical algorithms, PDI-CP and JDI-CP, and provide a
distribution-free coverage analysis under some commonly used detection and
imputation procedures. Notably, JDI-CP achieves a finite sample $1-2\alpha$
coverage guarantee. Numerical experiments on both synthetic and real datasets
demonstrate that our proposed algorithms exhibit robust coverage properties and
comparable efficiency to the oracle baseline.

### 3. [Dequantified Diffusion Schrödinger Bridge for Density Ratio Estimation](http://arxiv.org/pdf/2505.05034v1)

Authors: Wei Chen, Shigui Li, Jiacheng Li, Junmei Yang, John Paisley, Delu Zeng

Density ratio estimation is fundamental to tasks involving $f$-divergences,
yet existing methods often fail under significantly different distributions or
inadequately overlap supports, suffering from the \textit{density-chasm} and
the \textit{support-chasm} problems. Additionally, prior approaches yield
divergent time scores near boundaries, leading to instability. We propose
$\text{D}^3\text{RE}$, a unified framework for robust and efficient density
ratio estimation. It introduces the Dequantified Diffusion-Bridge Interpolant
(DDBI), which expands support coverage and stabilizes time scores via diffusion
bridges and Gaussian dequantization. Building on DDBI, the Dequantified
Schr\"odinger-Bridge Interpolant (DSBI) incorporates optimal transport to solve
the Schr\"odinger bridge problem, enhancing accuracy and efficiency. Our method
offers uniform approximation and bounded time scores in theory, and outperforms
baselines empirically in mutual information and density estimation tasks.

### 4. [A Two-Sample Test of Text Generation Similarity](http://arxiv.org/pdf/2505.05269v1)

Authors: Jingbin Xu, Chen Qian, Meimei Liu, Feng Guo

The surge in digitized text data requires reliable inferential methods on
observed textual patterns. This article proposes a novel two-sample text test
for comparing similarity between two groups of documents. The hypothesis is
whether the probabilistic mapping generating the textual data is identical
across two groups of documents. The proposed test aims to assess text
similarity by comparing the entropy of the documents. Entropy is estimated
using neural network-based language models. The test statistic is derived from
an estimation-and-inference framework, where the entropy is first approximated
using an estimation set, followed by inference on the remaining data set. We
showed theoretically that under mild conditions, the test statistic
asymptotically follows a normal distribution. A multiple data-splitting
strategy is proposed to enhance test power, which combines p-values into a
unified decision. Various simulation studies and a real data example
demonstrated that the proposed two-sample text test maintains the nominal Type
one error rate while offering greater power compared to existing methods. The
proposed method provides a novel solution to assert differences in document
classes, particularly in fields where large-scale textual information is
crucial.

### 5. [Clustering with Communication: A Variational Framework for Single Cell Representation Learning](http://arxiv.org/pdf/2505.04891v1)

Authors: Cong Qi, Yeqing Chen, Jie Zhang, Wei Zhi

Single-cell RNA sequencing (scRNA-seq) has revealed complex cellular
heterogeneity, but recent studies emphasize that understanding biological
function also requires modeling cell-cell communication (CCC), the signaling
interactions mediated by ligand-receptor pairs that coordinate cellular
behavior. Tools like CellChat have demonstrated that CCC plays a critical role
in processes such as cell differentiation, tissue regeneration, and immune
response, and that transcriptomic data inherently encodes rich information
about intercellular signaling. We propose CCCVAE, a novel variational
autoencoder framework that incorporates CCC signals into single-cell
representation learning. By leveraging a communication-aware kernel derived
from ligand-receptor interactions and a sparse Gaussian process, CCCVAE encodes
biologically informed priors into the latent space. Unlike conventional VAEs
that treat each cell independently, CCCVAE encourages latent embeddings to
reflect both transcriptional similarity and intercellular signaling context.
Empirical results across four scRNA-seq datasets show that CCCVAE improves
clustering performance, achieving higher evaluation scores than standard VAE
baselines. This work demonstrates the value of embedding biological priors into
deep generative models for unsupervised single-cell analysis.

### 6. [Learning Linearized Models from Nonlinear Systems under Initialization Constraints with Finite Data](http://arxiv.org/pdf/2505.04954v1)

Authors: Lei Xin, Baike She, Qi Dou, George Chiu, Shreyas Sundaram

The identification of a linear system model from data has wide applications
in control theory. The existing work that provides finite sample guarantees for
linear system identification typically uses data from a single long system
trajectory under i.i.d. random inputs, and assumes that the underlying dynamics
is truly linear. In contrast, we consider the problem of identifying a
linearized model when the true underlying dynamics is nonlinear, given that
there is a certain constraint on the region where one can initialize the
experiments. We provide a multiple trajectories-based deterministic data
acquisition algorithm followed by a regularized least squares algorithm, and
provide a finite sample error bound on the learned linearized dynamics. Our
error bound shows that one can consistently learn the linearized dynamics, and
demonstrates a trade-off between the error due to nonlinearity and the error
due to noise. We validate our results through numerical experiments, where we
also show the potential insufficiency of linear system identification using a
single trajectory with i.i.d. random inputs, when nonlinearity does exist.

### 7. [Enhancing the Dynamic Range of Quantum Sensing via Quantum Circuit Learning](http://arxiv.org/pdf/2505.04958v1)

Authors: Hideaki Kawaguchi, Yuichiro Mori, Takahiko Satoh, Yuichiro Matsuzaki

Quantum metrology is a promising application of quantum technologies,
enabling the precise measurement of weak external fields at a local scale. In
typical quantum sensing protocols, a qubit interacts with an external field,
and the amplitude of the field is estimated by analyzing the expectation value
of a measured observable. Sensitivity can, in principle, be enhanced by
increasing the number of qubits within a fixed volume, thereby maintaining
spatial resolution. However, at high qubit densities, inter-qubit interactions
induce complex many-body dynamics, resulting in multiple oscillations in the
expectation value of the observable even for small field amplitudes. This
ambiguity reduces the dynamic range of the sensing protocol. We propose a
method to overcome the limitation in quantum metrology by adopting a quantum
circuit learning framework using a parameterized quantum circuit to approximate
a target function by optimizing the circuit parameters. In our method, after
the qubits interact with the external field, we apply a sequence of
parameterized quantum gates and measure a suitable observable. By optimizing
the gate parameters, the expectation value is trained to exhibit a monotonic
response within a target range of field amplitudes, thereby eliminating
multiple oscillations and enhancing the dynamic range. This method offers a
strategy for improving quantum sensing performance in dense qubit systems.

### 8. [Boosting Statistic Learning with Synthetic Data from Pretrained Large Models](http://arxiv.org/pdf/2505.04992v1)

Authors: Jialong Jiang, Wenkang Hu, Jian Huang, Yuling Jiao, Xu Liu

The rapid advancement of generative models, such as Stable Diffusion, raises
a key question: how can synthetic data from these models enhance predictive
modeling? While they can generate vast amounts of datasets, only a subset
meaningfully improves performance. We propose a novel end-to-end framework that
generates and systematically filters synthetic data through domain-specific
statistical methods, selectively integrating high-quality samples for effective
augmentation. Our experiments demonstrate consistent improvements in predictive
performance across various settings, highlighting the potential of our
framework while underscoring the inherent limitations of generative models for
data augmentation. Despite the ability to produce large volumes of synthetic
data, the proportion that effectively improves model performance is limited.

### 9. [Local linear Fréchet curve regression in manifolds](http://arxiv.org/pdf/2505.05168v1)

Authors: M. D. Ruiz-Medina, A. Torres--Signes

Global Fr\'echet functional regression has been recently addressed from time
correlated bivariate curve data evaluated in a manifold (see Torres et al.
2025). For this type of curve data sets, the present paper solves the problem
of local linear approximation of the Fr\'echet conditional mean in an extrinsic
and intrinsic way. The extrinsic local linear Fr\'echet functional regression
predictor is obtained in the time varying tangent space by projection into an
orthornormal basis of the ambient Hilbert space. The conditions assumed ensure
the existence and uniqueness of this predictor, and its computation via
exponential and logarithmic maps. A weighted Fr\'echet mean approach is adopted
in the computation of an intrinsic local linear Fr\'echet functional regression
predictor. The asymptotic optimality of this intrinsic local approximation is
also proved. The performance of the empirical version of both, extrinsic and
intrinsic functional predictors, and of a Nadaraya-Watson type Fr\'echet curve
predictor is illustrated in the simulation study undertaken. The finite-sample
size properties are also tested in a real-data application via
cross-validation. Specifically, functional prediction of the magnetic vector
field from the time-varying geocentric latitude and longitude of the satellite
NASA's MAGSAT spacecraft is addressed.

### 10. [A Connection Between Learning to Reject and Bhattacharyya Divergences](http://arxiv.org/pdf/2505.05273v1)

Authors: Alexander Soen

Learning to reject provide a learning paradigm which allows for our models to
abstain from making predictions. One way to learn the rejector is to learn an
ideal marginal distribution (w.r.t. the input domain) - which characterizes a
hypothetical best marginal distribution - and compares it to the true marginal
distribution via a density ratio. In this paper, we consider learning a joint
ideal distribution over both inputs and labels; and develop a link between
rejection and thresholding different statistical divergences. We further find
that when one considers a variant of the log-loss, the rejector obtained by
considering the joint ideal distribution corresponds to the thresholding of the
skewed Bhattacharyya divergence between class-probabilities. This is in
contrast to the marginal case - that is equivalent to a typical
characterization of optimal rejection, Chow's Rule - which corresponds to a
thresholding of the Kullback-Leibler divergence. In general, we find that
rejecting via a Bhattacharyya divergence is less aggressive than Chow's Rule.

