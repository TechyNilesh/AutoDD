## Page 1

# Automatic Learning to Detect Concept Drift

Hang Yu¹, Tianyu Liu¹, Jie Lu¹ and Guangquan Zhang¹
¹Australian Artificial Intelligence Institute, University of Technology Sydney. Australia.
{hang.yu, Tianyu.Liu-1}@student.uts.edu.au, {Jie.Lu, guangquan.zhang}@uts.edu.au

## Abstract

Many methods have been proposed to detect concept drift, i.e., the change in the distribution of streaming data, due to concept drift causes a decrease in the prediction accuracy of algorithms. However, the most of current detection methods are based on the assessment of the degree of change in the data distribution, cannot identify the type of concept drift. In this paper, we propose Active Drift Detection with Meta learning (Meta-ADD), a novel framework that learns to classify concept drift by tracking the changed pattern of error rates. Specifically, in the training phase, we extract meta-features based on the error rates of various concept drift, after which a meta-detector is developed via a prototypical neural network by representing various concept drift classes as corresponding prototypes. In the detection phase, the learned meta-detector is fine-tuned to adapt to the corresponding data stream via stream-based active learning. Hence, Meta-ADD uses machine learning to learn to detect concept drifts and identify their types automatically, which can directly support drift understand. The experiment results verify the effectiveness of Meta-ADD.

&lt;watermark&gt;arXiv:2105.01419v1 [cs.AI] 4 May 2021&lt;/watermark&gt;

## 1 Introduction

Concept drift is a hot research topic in data stream mining [He et al., 2019], incremental learning [Losing et al., 2018], non-stationary learning [Pratama and Wang, 2019], and it can be further extended to machine learning. In the machine learning area, concept drift is a phenomenon in which the statistical properties of the predicted variable change over time in an arbitrary way [Lu et al., 2017]. This causes the predictions to become less accurate as time passes, and therefore needs to be detected. Figure 1 shows the four types of concept drift, categorized according to the changed pattern of data distribution [Gama et al., 2014].

In the two recent decades, although many methods of detecting concept drift have been proposed, they all, in general, follow a framework [Lu et al., 2019] which includes four steps: 1) data retrieval aims to retrieve data chunks from data streams; 2) data modeling aims to abstract the retrieved data and extract the key features; 3) test statistics calculation is the measurement of dissimilarity, or distance estimation [de Lima Cabral and de Barros, 2018]; 4) hypothesis testing uses a specific hypothesis test to evaluate the statistical significance of the change observed in step 3, or the p-value.

However, this framework faces two drawbacks: the first is the cold start problem. In current drift detection methods, an initial window is needed to collect the basic statistic properties for the hypothesis test. As a result, no detection strategies can be implemented in the initial window, but there may be concept drift in the initial window. The second is, to the best of our knowledge, the existing methods can only identify whether concept drift has occurred, but they are unable to identify what type of concept drift has occurred. The reason for this is that although the change of data distribution can be represented by test statistics such as average error rate, test statistics cannot represent the changed pattern of data distribution, i.e., the relationship between concept drift at adjacent timestamps cannot be captured. In practice, knowing the type of concept drift is useful to understand them. For example, if a person’s weight suddenly decreases, this may indicate he has health problems which should be addressed instantly, but if a person’s weight gradual decreases over a long time, such as over one year, the situation may be normal because the man is making an effort to lose weight.

In this paper, we propose a novel framework to learn a meta-detector with the ability to classify concept drift which can be trained in advance. When a data stream is inputted, it can be used to directly classify concept drift without setting an initial window. However, this framework faces two main

&lt;img&gt;
A figure showing four types of concept drift:
1. Sudden Drift: A bar chart with blue bars representing the initial data distribution, followed by orange bars representing a sudden change in the data distribution.
2. Gradual Drift: A bar chart with blue bars representing the initial data distribution, followed by orange bars representing a gradual change in the data distribution.
3. Incremental Drift: A bar chart with blue bars representing the initial data distribution, followed by orange bars representing an incremental change in the data distribution.
4. Reoccurring context: A bar chart with blue bars representing the initial data distribution, followed by orange bars representing a reoccurring change in the data distribution.
The y-axis is labeled "Data distribution" and the x-axis is labeled "Time".
&lt;/img&gt;
Figure 1: Types of concept drift.

---


## Page 2

challenges. On the one hand, although concept drift can be classified into four types, the presentations, i.e., changed pattern of data distribution, of one type of drifts is infinite possibly over 1000000. By contrast, the presentations of this type of drifts, which are included in training data, are very small possibly 1000. On the other hand, for each type of concept drift, each data stream may have its own unique presentation, so the meta-detector must have the ability to be able to adjust to fit the detected data stream. To address these two challenges, we propose Active Drift Detection with Meta learning (Meta-ADD). The contributions of this paper are summarized as follows:

*   We propose a framework that learns to detect drift by learning the presentation of each type of concept drift. As a result, in our framework, not only can concept drift be detected, the type of drift can also be identified to help us understand them.
*   We propose to transform the pre-training of a meta-detector into a few-shot learning problem, and thereby proposing a meta-learning method [Finn et al., 2017] based on prototypical-networks [Snell et al., 2017] to learn the meta-detector. In this meta-detector, various concept drift classes can be represented as a corresponding single prototype.
*   To improve the accuracy of detecting concept drift, we propose a stream-based active learning algorithm (SAL) which is capable of handling various concept drifts by adapting the meta-detector to the underlying distribution in the stream.

The rest of this paper is organized as follows. Section 2 presents the related work. Section 3 provides the details of our proposed Meta-ADD framework. Section 4 discusses the experiment results for several well-known datasets in this research area. Finally, Section 5 concludes the paper and presents the future work.

## 2 Related work

The survey of the literature related to drift detection methods reveals that in general, these can be divided into three categories:

### 2.1 Error Rate-based Drift Detection

Error rate-based drift detection methods form the largest category of drift detection algorithms. The intuition behind this type of algorithms is to monitor fluctuations in the error rate as time passes. Once the change of the error rate is proven to be statistically significant, a drift alarm will be triggered. One of the top referenced this type of algorithms is the Drift Detection Method (DDM) [Gama et al., 2004] which was the first algorithm to contain defined warning and drift levels for signalling concept drift. Many subsequent algorithms have adopted a similar implementation, e.g., Early Drift Detection Method (EDDM) [Baena-Garcia et al., 2006], Hoeffding’s inequality-based Drift Detection Method (HDDM) [Frías-Blanco et al., 2015] and based on the Kolmogorov-Smirnov (KS) statistical test. In contrast to methods such as DDM, a two-time window-based drift detection algorithm called Adaptive Windowing (ADWIN) [Bifet and Gavalda, 2007] was proposed. In ADWIN, the size of the compared windows can be adjusted automatically.

### 2.2 Data Distribution-based Drift Detection

The second largest category of drift detection algorithms is data distribution-based drift detection. In this category of algorithms, the dissimilarity between the distribution of new data and historical data is quantified by using a distance metric. Once the dissimilarity is proven to be statistically significantly different, a drift alarm will be triggered. According to the literature, the first formal treatment of change detection in data streams was proposed by [Kifer et al., 2004]. Another typical data distribution-based drift detection algorithm is the Information-Theoretic Approach (ITA) [Dasu et al., 2006]. Similar implementations have been adopted in the Competence Model-based drift detection (CM) [Lu et al., 2014], Equal Density Estimation (EDE) [Yu et al., 2020], and Local Drift Degree-based Density Synchronized Drift Adaptation (LDD-DSDA) [Liu et al., 2017].

### 2.3 Multiple Hypothesis Test Drift Detection

Multiple hypothesis test drift detection algorithms apply similar techniques to those mentioned in the previous two categories. The novelty of this type of algorithms is that the multiple hypothesis tests is used to detect concept drift in different ways. For example, the Just-In-Time adaptive classifier (JIT) [Alippi and Roveri, 2008] is the first algorithm to set multiple drift detection hypotheses in this way, but Hierarchical Change-Detection Tests (HCDTs) [Alippi et al., 2017] is the first attempt to address concept drift using a hierarchical architecture.

## 3 Active Drift Detection with Meta-learning

In this section, we elaborate on the Active Drift Detection with Meta learning (Meta-ADD). Meta-ADD includes two main phases: training phase and detection phase. During the training phase, we extract the meta-features of various concept drifts, and then learn a meta-detector where various concept drift classes can be represented as a corresponding single prototype [Shao et al., 2018]. In the detection phase, the learned meta-detector is used to detect concept drift and is further fine-tuned via stream-based active learning [Krawczyk and Cano, 2019].

### 3.1 Extracting Meta-Features

In this subsection, we describe the extraction of the meta-features that can be used across different datasets. Although there are four types of concept drift, the reoccurring concept type focuses on tracking whether an old concept occurs again. In contrast, our work focuses on detecting concept drift immediately instead of considering whether a concept is an old concept. So we did not use the reoccurring concept type and instead used a concept type we named normal to indicate that there was no concept drift. It is proved that a change in the data distribution can be represented by a change of error rates, so intuitively, the change of error rate is critical for deciding which type of concept drift has occurred:

---


## Page 3

*   **Sudden drift:** the error rate will decrease sharply within a short time.
*   **Gradual drift:** the error rate will fluctuate over a period of time.
*   **Incremental drift:** the error rate will increase incrementally over a period of time.
*   **Normal:** the error rate will remain stable.

Based on the intuitions above, we empirically extract the gap between the average error rates of two windows as meta-features. Specifically, for each type of concept drift, we first generate $N$ number of data streams. A prediction model such as Neural network, decision tree or SVR was directly applied to a data stream to obtain error rate $e_t$ at each timestamp $t$. In addition, we use a window $W_i$ with $n$ length, that is, there are $n$ number of samples in $W_i$, so the average error rates of $W_i$ are calculated as:

$$\hat{e}_i = \frac{\sum_{t=1}^{n} e_i}{n} \quad (1)$$

Furthermore, the gap between the average error rates of two windows is calculated as:

$$Gap_i = \hat{e}_{i+1} - \hat{e}_i \quad (2)$$

Assume there are $m = l \times n$ samples in each data stream, so $l$ number of windows can be obtained, and thereby $l - 1$ is the number of $Gap_i$ which can be obtained. Hence, a training sample in meta-training is represented as $G_i : \{\hat{X} \times \hat{y}\} \in \mathbb{R}^l$, where $\hat{X} = \{Gap_1, \ldots, Gap_{l-1}\}$, and $\hat{y} \in 1, \ldots, K$ is the corresponding labels, and thereby an original dataset with $N$ number of data streams, i.e., $g : \{X \times y\}^{N \times m}$, is mapped to $G : \{\hat{X} \times \hat{y}\} \in \mathbb{R}^{N \times l}$, such that $G$ is less dependent on the dataset.

Note that our framework allows flexibility in the choice of features. For example, we may be able to improve the performance by using different predicted models or a more intelligent windowing strategy. To make our contribution focused, we adopt simple models and tumbling windows in all our experiments, which results in a reasonable performance based on our empirical results. How we can better model the transferable information will be interesting future work to enhance the Meta-ADD framework.

## 3.2 Meta-detector via a Prototypical Neural Network

In this subsection, a prototypical neural network is applied to extract meta features from various data streams. The concept drift detection issue is considered to be a classification task by learning a neural network for mapping the data stream into an embedding space and extracting prototypes for various concept drift categories to be the mean vector of the embedded support data stream. The detection of concept drift is performed for the embedded data stream by selecting the nearest concept drift prototype.

Based on the support dataset $G$, the prototypical neural network consists of two steps, as illustrated in Figure 2. First, an em embedding neural network $f_\theta : \mathbb{R}^l \rightarrow \mathbb{R}^M$ with the learnable parameter $\theta$ maps the stream data $X_i$ into the embedding space $f_\theta(X_i)$, then a single prototype representation $c_k$ for each class $k$ can be taken as the mean of its support set in the embedding space:

$$c_k = \frac{1}{|S_k|} \sum_{(X_i, y_i) \in R} f_\theta(\hat{X}_i)$$

where $S_k$ is the set of stream data with label $k$. Given a distance function $d : \mathbb{R}^M \times \mathbb{R}^M \rightarrow \mathbb{R}$, eg. cosine similarity, classification is then performed for a query point $X$ by finding the nearest prototype $c_k$ over a distance in the embedding space:

$$p_\theta(\hat{y} = k \mid \hat{x}) = \frac{\exp(-d(f_\theta(\hat{x}), c_k))}{\sum_{k'} \exp(-d(f_\theta(\hat{x}), c_{k'}))}$$

We then use negative log-likelihood function $J(\theta) = -\log p_\theta(y = k \mid \hat{x})$ as the training objective. During the training phase, $N_s$ examples are selected randomly for each concept drift class to represent the prototype in the embedding space, then $N_q$ samples are chosen randomly as the query set to measure the estimation accuracy of the learned meta learner. The pseudocode to train the model is provided in Algorithm 1.

## 3.3 Active Drift Detection

Although a meta-detector can be obtained in the training process and can be directly applied to detect concept drift, the representation of each type of concept drift on different data streams can vary and each has unique characteristics. Hence, when the meta-detector is applied to a data stream, it must be able to adapt to this data stream to achieve better performance.

To update the meta-detector, when we collect a sample $S_i = \{e_{i-i}, \ldots e_i\}$ where $e_i$ is the obtained error rate at timestamp $i$, we need to know the true type of concept drift which has occurred on this sample to update the meta-detector. However, unfortunately, the true type of concept

```mermaid
graph TD
    subgraph Input Data
        A[X] --> B[Y]
    end
    subgraph Model
        B --> C[Z]
    end
    subgraph Output
        C --> D[P]
    end
```

```mermaid
graph TD
    A[Input: training data steam G : {(X₁, y₁), ..., (Xₗ, yₗ)}, where the corresponding label yᵢ ∈ 1, ..., K, Dₖ is the set of stream data with label k.] --> B[Parameter: the number of support data stream Nₛ, the number of query data stream Nₚ, the number of concept drift class K.]
    B --> C[Output: the training loss of J(θ)]
    C --> D[for k in {1, ..., K}]
    D --> E[Sₖ ← RANDOMSAMPLE(Dₖ, Nₛ)]
    E --> F[Qₖ ← RANDOMSAMPLE(Dₖ, Nₚ)]
    F --> G[cₖ = 1/Nₛ Σ(Xᵢ, yᵢ) f_θ(Xᵢ)]
    G --> H[end for]
    H --> I[J ← 0]
    I --> J[for k in {1, ..., Nₚ}]
    J --> K[for (x, y) in Qₖ]
    K --> L[J ← J - 1/KNₚ log exp(-d(f_θ(x), cₖ)) / Σk' exp(-d(f_θ(x), cₖ'))]
    L --> M[end for]
    M --> N[end for]

---


## Page 4

mermaid
graph TD
    subgraph Input Data Streams
        A[Data Stream 1]
        B[Data Stream 2]
        C[Data Stream 3]
        D[Data Stream 4]
    end

    subgraph Prototypical Neural Network
        E[f(θ)]
    end

    subgraph Embedding and Clustering
        F[ ]
        G[ ]
        H[ ]
        I[ ]
        J[ ]
        K[ ]
        L[ ]
        M[ ]
        N[ ]
        O[ ]
        P[ ]
        Q[ ]
        R[ ]
        S[ ]
        T[ ]
        U[ ]
        V[ ]
        W[ ]
        X[ ]
        Y[ ]
        Z[ ]
    end

    subgraph Classification
        AA[f(θ)]
    end

    subgraph Output Classifications
        BB[ ]
        CC[ ]
        DD[ ]
        EE[ ]
        FF[ ]
        GG[ ]
        HH[ ]
        II[ ]
        JJ[ ]
        KK[ ]
        LL[ ]
        MM[ ]
        NN[ ]
        OO[ ]
        PP[ ]
        QQ[ ]
        RR[ ]
        SS[ ]
        TT[ ]
        UU[ ]
        VV[ ]
        WW[ ]
        XX[ ]
        YY[ ]
        ZZ[ ]
    end

    A --> E
    B --> E
    C --> E
    D --> E

    E --> F
    E --> G
    E --> H
    E --> I
    E --> J
    E --> K
    E --> L
    E --> M
    E --> N
    E --> O
    E --> P
    E --> Q
    E --> R
    E --> S
    E --> T
    E --> U
    E --> V
    E --> W
    E --> X
    E --> Y
    E --> Z

    F --> AA
    G --> AA
    H --> AA
    I --> AA
    J --> AA
    K --> AA
    L --> AA
    M --> AA
    N --> AA
    O --> AA
    P --> AA
    Q --> AA
    R --> AA
    S --> AA
    T --> AA
    U --> AA
    V --> AA
    W --> AA
    X --> AA
    Y --> AA
    Z --> AA

    AA --> BB
    AA --> CC
    AA --> DD
    AA --> EE
    AA --> FF
    AA --> GG
    AA --> HH
    AA --> II
    AA --> JJ
    AA --> KK
    AA --> LL
    AA --> MM
    AA --> OO
    AA --> PP
    AA --> QQ
    AA --> RR
    AA --> SS
    AA --> TT
    AA --> UU
    AA --> VV
    AA --> WW
    AA --> XX
    AA --> YY
    AA --> ZZ
```

Figure 2: The prototypical neural network. (The left side is the whole structure of prototypical neural network, which consists of embedding and clustering phase; in the right side the new sample is classified by finding the nearest prototype over a distance in the embedding space)

```mermaid
graph TD
    subgraph Data stream
        A[...]
        B[...]
        C[...]
        D[...]
    end

    subgraph Label
        E[Normal]
        F[Gradual Drift]
        G[Incremental Drift]
        H[Sudden Drift]
    end

    subgraph Updating
        I[Updating]
    end

    subgraph Training
        J[Training]
    end

    subgraph Detecting
        K[Detecting]
    end

    subgraph Labeling
        L[Labelling]
    end

    subgraph Analyst
        M[Analyst]
    end

    subgraph Uncertain label
        N[Uncertain label]
    end

    subgraph Classification
        O[Classification]
    end

    subgraph Meta-detector
        P[Meta-detector]
    end

    subgraph Embedding
        Q[Embedding]
    end

    A --> B
    B --> C
    C --> D
    D --> A

    E --> F
    F --> G
    G --> H
    H --> E

    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> I
```

Figure 3: Structure of Meta-ADD algorithm.

drift is unknown. Although the true type can be inputted manually (label cost), the label cost is too expensive if the true label is manually input as soon as sample $S_i$ is collected. Hence, only the most valuable samples for labeling to update the meta-detector are used.

In Meta-ADD, the meta-detector denotes the probability $P(c)$ that sample $S_i$ belongs to class $c$, where $\sum_{c \in C} P(c) = 1$. Then, the entropy of classification can be calculated as:

$$H(C) = -\sum_{c \in C} P(c) \log p(c) \quad (1)$$

The entropy means the uncertainty of classification. A larger entropy means the type of this concept drift is more uncertain, and therefore this type of concept drift needs to be labelled manually.

The structure of the proposed Meta-ADD algorithm is shown in Figure 3.

## 4 Experiments

To evaluate the Meta-ADD, we conducted four experiments. The first two experiments described in Section 4.1 were designed to analyze the effectiveness of Meta-ADD in identifying the type of concept drift which has occurred. The third experiment described in Section 4.2 compares the difference in the performance of other drift detection algorithms and Meta-ADD on toy datasets with known drifts. The fourth experiment described in Section 4.3 evaluates the strengths and weaknesses of Meta-ADD on real-world datasets with unknown drifts. All experiments are implemented in Python 3.7 version on Windows 10 running on a PC with system configuration Intel Core i7 processor (2.80 GHz) with 16-GB RAM.

### 4.1 Demonstration of Meta-ADD

#### Experiment 1. Drift detection on data streams with different meta-learning structures

In this experiment, we focus on investigating how meta-learning impacts learning performance, because, in our proposed Meta-ADD, the quality of the detector depends on the results of meta-learning, and the results of meta-learning depend on the structure of prototypical-networks. Hence, the intuition is to generate a set of data streams with predefined types of concept drift, and then observe the results when Meta-ADD with different types of prototypical-networks and parameters is applied to those streams. We expect to see that meta-learning enables our detector to classify concept drift.

**Experiment Settings.** First, based on Python package sk-multiflow [Montiel et al., 2018] (version 0.5.3), we generate a total of 8000 data streams, and each data stream has 2000 instances to simulate the prequential prediction errors of a learning model. Furthermore, each data stream only has one type of drift. Then, we select 1250 data streams from each type of concept drift, i.e., a total of 5000 data streams, as an original training dataset to extract meta-features. The remaining data streams are mixed as testing datasets. GaussianNB is imported from sklearn [Pedregosa et al., 2011] as the default learning model because it has a fast-processing speed and a partial fit option. As for the structures of the prototypical neural network, two embedding functions: a 4-layered fully connected neural network and recurrent neural network are selected as the nonlinear mapping. In the training phase, for each concept drift category, 5 support data streams are chosen randomly to act as the prototype and 15 query data streams are selected to evaluate the performance of the model. During the testing phase, 20 support data streams are randomly selected from the training dataset to perform as the final prototype. For all the experiments, ReLU activations are applied

---


## Page 5

and the distance function is the cosine similarity function. All our models are trained via SGD with Adam, with a learning rate of $10^{-3}$.

<table>
  <tr>
    <th>Type</th>
    <th>Network</th>
    <th>Acc</th>
  </tr>
  <tr>
    <td>Sudden</td>
    <td>FCN</td>
    <td>98.5%</td>
  </tr>
  <tr>
    <td></td>
    <td>RNN</td>
    <td>32.3%</td>
  </tr>
  <tr>
    <td>Gradual</td>
    <td>FCN</td>
    <td>94.2%</td>
  </tr>
  <tr>
    <td></td>
    <td>RNN</td>
    <td>53.4%</td>
  </tr>
  <tr>
    <td>Incremental</td>
    <td>FCN</td>
    <td>91.3%</td>
  </tr>
  <tr>
    <td></td>
    <td>RNN</td>
    <td>52.1%</td>
  </tr>
  <tr>
    <td>Normal</td>
    <td>FCN</td>
    <td>93.8%</td>
  </tr>
  <tr>
    <td></td>
    <td>RNN</td>
    <td>82.2%</td>
  </tr>
</table>

Table 1: Accuracy of different deep structures. (Acc is the abbreviation of accuracy; 50-25 means the number of meta-features is 50 and the sampling frequency is 25.)

**Findings and Discussion.** The results of Experiment 1 are shown in Table 1. From Table 1, we can see Meta-ADD was able to identify the simulated concept drifts and, overall, the results were in line with expectations, in that the FCN achieves the better performance because the relationship between samples in a data stream is not required strong. In addition, we can see each type of concept drift, with same settings, performed differently.

**Experiment 2. Drift Detection on data streams with different meta-features**

In this experiment, we focus on investigating how the meta-features impact learning performance. However, in our proposed Meta-ADD, extracting meta-features depends on two parameters $l$ and $n$, where $l$ means the number of meta-features, $n$ means the sampling frequency. Hence, the intuition is to generate a set of data streams with a predefined type of concept drift, and then observe the results when Meta-ADD with different $l$ and $n$ is applied to these streams. We expect to see Meta-ADD with different $l$ and $n$ perform differently on classifying concept drifts.

**Experiment Settings.** We again chose GaussianNB as the default learning model and the same parameter settings as in Experiment 1. Based on the results of experiment 1, the deep structure of meta-learning is implemented on FCN. However, to explore the influence of $l$ and $n$ when extracting the meta-features, the number of meta-features $l$ was changed within the set $l_{valid} \in \{50, 100, 200\}$, and the sampling frequency $n$ was also changed within the set $n_{valid} \in \{1, 25, 50\}$.

<table>
  <tr>
    <th></th>
    <th>$n = 1$</th>
    <th>$n = 25$</th>
    <th>$n = 50$</th>
  </tr>
  <tr>
    <td>$l = 50$</td>
    <td>71.92%</td>
    <td>86.45%</td>
    <td>92.57%</td>
  </tr>
  <tr>
    <td>$l = 100$</td>
    <td>73.83%</td>
    <td>72.53%</td>
    <td>94.56%</td>
  </tr>
  <tr>
    <td>$l = 200$</td>
    <td>70.42%</td>
    <td>72.98%</td>
    <td>90.68%</td>
  </tr>
</table>

Table 2: Accuracy of different meta-features

**Findings and Discussion.** The results of Experiment 2 are plotted in Table 2. From Table 2, two interesting points are worth mentioning: first is we can see that $l$ and $n$ influence the prediction accuracy. However, the relationship between the accuracy of the drift detection, the number of meta-features $l$, and the sampling frequency $n$, respectively, is not the same. When the $n$ increases, the prediction accuracy increases in general. In contrast, when the $l$ increases, the prediction accuracy may decreases.

**4.2 Stream Learning on Toy Datasets with Simulated Concept Drift**

**Experiment 3. Drift Detection on Toy Datasets**

In this experiment, we focus on investigating the performance of Meta-ADD compared to current drift detection methods. However, current drift detection methods cannot classify concept drift, so we only compare the prediction error based on extreme cases. We generate a set of data streams with a pre-defined type of drift, drift frequencies and severities to simulate the prequential prediction errors of a learning model with different drift detection methods. If there are less prediction errors, this means that the performance of the drift detection method is better. We expect to see that the learning model with our proposed meta-ADD achieves the best performance on different data streams with different concept drifts.

**Experiment Settings.** GaussianNB is again choses as the default learning model. The toy datasets were generated by skmultiflow’s built-in data generation functions. We chose SEA generator, Rotating Hyperplane generator, AGR (AGRRAWAL) generator, RTG (Random Tree Generator) and RBF generator. All generators generate 10000 samples. The compared algorithms were ADWIN, DDM, EDDM, HDDMA, HDDM-W, Page-Hinkley, KSWIN implemented in skmultiflow with their advised drift thresholds. We use the parameters with the best performance in experiment 1 to train the model and set $l$ to 100 and $n$ to 1 to extract the meta-features.

**Findings and Discussion.** The results of Experiment 3 are shown in Table 3. In general, the prediction accuracy of Meta-Add is better than the other drift detection methods over all data streams. In addition, Table 3 also shows Meta-ADD has the largest F1 score, i.e., detecting concept drift by Meta-ADD is more authentic. Thus, our proposed Meta-ADD achieves better performance than current drift detection methods. A comparison of the two last lines of Table 3 shows that the performance of Meta-ADD was improved by active learning.

**4.3 Stream Learning on Real-world datasets**

**Experiment 4. Drift Detection on Real-word Datasets**

This experiment investigates whether our proposed Meta-ADD continues to obtain the best performance on real-world concept drift evaluation datasets. We expect to see similar results to those of Experiment 3.

**Experiment Settings.** The basic settings are the same as Experiment 3. The main difference is that we used a group of real-world datasets with unknown drift, and these were most often used to evaluate the performance of drift detection methods. These datasets were [Bifet et al., 2010]:

*   **Elec:** the Electricity Price Dataset has 45325 samples and each sample has 9 features

---


## Page 6

<table>
<thead>
<tr>
<th></th>
<th colspan="2">SEA</th>
<th colspan="2">HYP</th>
<th colspan="2">AGR</th>
<th colspan="2">RBF</th>
<th colspan="2">RTG</th>
</tr>
<tr>
<th></th>
<th>Acc</th>
<th>F1</th>
<th>Acc</th>
<th>F1</th>
<th>Acc</th>
<th>F1</th>
<th>Acc</th>
<th>F1</th>
<th>Acc</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>*</td>
<td>83.7%</td>
<td>Nan</td>
<td>81.6%</td>
<td>Nan</td>
<td>70.3%</td>
<td>Nan</td>
<td>69.3%</td>
<td>Nan</td>
<td>64.5%</td>
<td>Nan</td>
</tr>
<tr>
<td>*+ADWIN</td>
<td>84.3%</td>
<td>0.62</td>
<td>81.6%</td>
<td>0.00</td>
<td>71.3%</td>
<td>0.50</td>
<td>69.3%</td>
<td>0.00</td>
<td>64.5%</td>
<td>0.00</td>
</tr>
<tr>
<td>*+DDM</td>
<td>83.7%</td>
<td>0.00</td>
<td>86.1%</td>
<td>0.63</td>
<td>71.2%</td>
<td>0.50</td>
<td>71.5%</td>
<td>0.66</td>
<td>64.5%</td>
<td>0.00</td>
</tr>
<tr>
<td>*+EDDM</td>
<td>84.0%</td>
<td>0.38</td>
<td>86.1%</td>
<td>0.63</td>
<td>65.1%</td>
<td>0.25</td>
<td>67.8%</td>
<td>0.38</td>
<td>65.2%</td>
<td>0.38</td>
</tr>
<tr>
<td>*+HDDM-A</td>
<td>84.7%</td>
<td>0.69</td>
<td>81.6%</td>
<td>0.00</td>
<td>72.3%</td>
<td>0.63</td>
<td>69.3%</td>
<td>0.00</td>
<td>64.5%</td>
<td>0.00</td>
</tr>
<tr>
<td>*+HDDM-W</td>
<td>84.5%</td>
<td>0.63</td>
<td>86.4%</td>
<td>0.70</td>
<td>72.4%</td>
<td>0.63</td>
<td>69.8%</td>
<td>0.50</td>
<td>66.3%</td>
<td>0.55</td>
</tr>
<tr>
<td>*+Page-Hinkley</td>
<td>85.0%</td>
<td>0.71</td>
<td>81.6%</td>
<td>0.00</td>
<td>71.4%</td>
<td>0.50</td>
<td>69.3%</td>
<td>0.00</td>
<td>64.5%</td>
<td>0.00</td>
</tr>
<tr>
<td>*+KSWIN</td>
<td>84.1%</td>
<td>0.38</td>
<td>86.2%</td>
<td>0.66</td>
<td>71.5%</td>
<td>0.50</td>
<td>69.9%</td>
<td>0.63</td>
<td>65.2%</td>
<td>0.38</td>
</tr>
<tr>
<td>*+Meta-DD</td>
<td>86.1%</td>
<td>0.77</td>
<td>87.8%</td>
<td>0.83</td>
<td>73.2%</td>
<td>0.69</td>
<td>72.2%</td>
<td>0.71</td>
<td>67.4%</td>
<td>0.70</td>
</tr>
<tr>
<td>*+Meta-ADD</td>
<td>89.2%</td>
<td>0.85</td>
<td>91.1%</td>
<td>0.89</td>
<td>74.6%</td>
<td>0.79</td>
<td>73.7%</td>
<td>0.74</td>
<td>68.9%</td>
<td>0.75</td>
</tr>
</tbody>
</table>

* represents the GaussianNB algorithm. Nan means null value, and 0.0 indicates that no drifts have been detected.

Table 3: Experiment results on toy datasets

<table>
<thead>
<tr>
<th></th>
<th colspan="2">Elec</th>
<th colspan="2">Weather</th>
<th colspan="2">Spam</th>
<th colspan="2">Air</th>
<th colspan="2">Poke</th>
</tr>
<tr>
<th></th>
<th>Acc</th>
<th>DN</th>
<th>Acc</th>
<th>DN</th>
<th>Acc</th>
<th>DN</th>
<th>Acc</th>
<th>DN</th>
<th>Acc</th>
<th>DN</th>
</tr>
</thead>
<tbody>
<tr>
<td>*</td>
<td>73.9%</td>
<td>0</td>
<td>68.7%</td>
<td>0</td>
<td>89.3%</td>
<td>0</td>
<td>58.1%</td>
<td>0</td>
<td>51.2%</td>
<td>0</td>
</tr>
<tr>
<td>*+ADWIN</td>
<td>79.1%</td>
<td>46</td>
<td>69.5%</td>
<td>8</td>
<td>92.4%</td>
<td>3</td>
<td>62.2%</td>
<td>70</td>
<td>54.2%</td>
<td>1</td>
</tr>
<tr>
<td>*+DDM</td>
<td>82.1%</td>
<td>117</td>
<td>70.3%</td>
<td>3</td>
<td>91.9%</td>
<td>8</td>
<td>61.9%</td>
<td>178</td>
<td>54.2%</td>
<td>1</td>
</tr>
<tr>
<td>*+EDDM</td>
<td>82.6%</td>
<td>115</td>
<td>73.7%</td>
<td>49</td>
<td>93.4%</td>
<td>13</td>
<td>61.5%</td>
<td>122</td>
<td>54.5%</td>
<td>4</td>
</tr>
<tr>
<td>*+HDDM-A</td>
<td>83.0%</td>
<td>123</td>
<td>72.3%</td>
<td>29</td>
<td>92.1%</td>
<td>4</td>
<td>62.6%</td>
<td>617</td>
<td>52.0%</td>
<td>604</td>
</tr>
<tr>
<td>*+HDDM-W</td>
<td>82.7%</td>
<td>99</td>
<td>72.6%</td>
<td>38</td>
<td>92.2%</td>
<td>2</td>
<td>61.5%</td>
<td>63</td>
<td>54.1%</td>
<td>15</td>
</tr>
<tr>
<td>*+Page-Hinkley</td>
<td>78.2%</td>
<td>17</td>
<td>69.6%</td>
<td>5</td>
<td>91.8%</td>
<td>1</td>
<td>62.8%</td>
<td>664</td>
<td>53.6%</td>
<td>306</td>
</tr>
<tr>
<td>*+KSWIN</td>
<td>81.4%</td>
<td>91</td>
<td>67.8%</td>
<td>45</td>
<td>92.4%</td>
<td>5</td>
<td>61.3%</td>
<td>155</td>
<td>55.7%</td>
<td>267</td>
</tr>
<tr>
<td>*+Meta-DD</td>
<td>83.5%</td>
<td>125</td>
<td>74.0%</td>
<td>50</td>
<td>94.5%</td>
<td>12</td>
<td>63.9%</td>
<td>136</td>
<td>56.1%</td>
<td>202</td>
</tr>
<tr>
<td>*+Meta-ADD</td>
<td>85.7%</td>
<td>101</td>
<td>76.1%</td>
<td>55</td>
<td>95.2%</td>
<td>11</td>
<td>64.7%</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

DN represents the number of detected drifts.

Table 4: Experiment results on real-world datasets

*   **Weather:** the NAOO Weather Dataset has 18712 samples and each sample has 9 features.
*   **Spam:** the Spam Filtering Dataset has 9829 samples and each sample has 501 features.
*   **Air:** the Airline Delay Dataset has 539395 samples and each sample has 8 features.
*   **Poke:** the Poker Hand Dataset has 923720 samples and each sample has 11 features.

In the Elec, Weather, Spam datasets, parameters $l$ and $n$ were set to 100 and 1, respectively. In the remaining datasets, parameters $l$ and $n$ were set to 100 and 25, respectively.

**Findings and Discussion.** The learning accuracies of the different drift detection methods appear in Table 4. From Table 4, we can see Meta-ADD continues to obtain the best performance on all datasets. In addition, we compared the number of drifts detected using different settings of parameters $l$ and $n$. In general, with an increase in $l$ and $n$, the number of detected drifts decreases. For example, in the Elec dataset, when parameter $l = 50$, but $n$ increases from 1 to 25, the number of detected drifts decreases from 158 to 28.

## 5 Conclusion

In this paper, we propose a novel framework named Meta-ADD to learn a meta-detector for detecting concept drift. Different to the current drift detection methods, Meta-ADD does not rely on rules set by humans to judge whether concept drift has occurred, rather it proves that machines can be offline trained to classify concept drift, and thereby solve problems facing the existing current drift detection methods, i.e., cold start and an inability to classify concept drift. Hence, compared with current methods, using Meta-ADD to detect concept drift can better understand them. Extensive experimental results also show the effectiveness of Meta-ADD.

---


## Page 7

# References

[Alippi and Roveri, 2008] Cesare Alippi and Manuel Roveri. Just-in-time adaptive classifiers—part i: Detecting nonstationary changes. *IEEE Transactions on Neural Networks*, 19(7):1145–1153, 2008.

[Alippi et al., 2017] C. Alippi, G. Boracchi, and M. Roveri. Hierarchical change-detection tests. *IEEE Transactions on Neural Networks and Learning Systems*, 28(2):246–258, 2017.

[Baena-García et al., 2006] Manuel Baena-García, José del Campo-Ávila, Raúl Fidalgo, Albert Bifet, R Gavalda, and R Morales-Bueno. Early drift detection method. In *Fourth international workshop on knowledge discovery from data streams*, volume 6, pages 77–86, 2006.

[Bifet and Gavalda, 2007] Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM international conference on data mining*, pages 443–448. SIAM, 2007.

[Bifet et al., 2010] Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Philipp Kranen, Hardy Kremer, Timm Jansen, and Thomas Seidl. Moa: Massive online analysis, a framework for stream classification and clustering. In *Proceedings of the First Workshop on Applications of Pattern Analysis*, pages 44–50. PMLR, 2010.

[Dasu et al., 2006] Tamraparni Dasu, Shankar Krishnan, Suresh Venkatasubramanian, and Ke Yi. An information-theoretic approach to detecting changes in multidimensional data streams. In *In Proc. Symp. on the Interface of Statistics, Computing Science, and Applications*. Citeseer, 2006.

[de Lima Cabral and de Barros, 2018] Danilo Rafael de Lima Cabral and Roberto Souto Maior de Barros. Concept drift detection based on fisher’s exact test. *Information Sciences*, 442:220–234, 2018.

[Finn et al., 2017] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning-Volume 70*, pages 1126–1135, 2017.

[Frías-Blanco et al., 2015] I. Frías-Blanco, J. d. Campo-Ávila, G. Ramos-Jiménez, R. Morales-Bueno, A. Ortiz-Díaz, and Y. Caballero-Mota. Online and non-parametric drift detection methods based on hoeffding’s bounds. *IEEE Transactions on Knowledge and Data Engineering*, 27(3):810–823, 2015.

[Gama et al., 2004] Joao Gama, Pedro Medas, Gladys Castillo, and Pedro Rodrigues. Learning with drift detection. In *Brazilian symposium on artificial intelligence*, pages 286–295. Springer, 2004.

[Gama et al., 2014] João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia. A survey on concept drift adaptation. *ACM computing surveys (CSUR)*, 46(4):1–37, 2014.

[He et al., 2019] Yi He, Baijun Wu, Di Wu, Ege Beyazit, Sheng Chen, Xindong Wu, et al. Online learning from capricious data streams: A generative approach. In *IJCAI*, pages 2491–2497, 2019.

[Kifer et al., 2004] Daniel Kifer, Shai Ben-David, and Johannes Gehrke. Detecting change in data streams. In *VLDB*, volume 4, pages 180–191. Toronto, Canada, 2004.

[Krawczyk and Cano, 2019] Bartosz Krawczyk and Alberto Cano. Adaptive ensemble active learning for drifting data stream mining. In *IJCAI*, pages 2763–2771, 2019.

[Liu et al., 2017] Anjin Liu, Yiliao Song, Guangquan Zhang, and Jie Lu. Regional concept drift detection and density synchronized drift adaptation. In *IJCAI International Joint Conference on Artificial Intelligence*, 2017.

[Losing et al., 2018] Viktor Losing, Barbara Hammer, and Heiko Wersing. Incremental on-line learning: A review and comparison of state of the art algorithms. *Neurocomputing*, 275:1261–1274, 2018.

[Lu et al., 2014] Ning Lu, Guangquan Zhang, and Jie Lu. Concept drift detection via competence models. *Artificial Intelligence*, 209:11–28, 2014.

[Lu et al., 2017] Yang Lu, Yiu-ming Cheung, and Yuan Yan Tang. Dynamic weighted majority for incremental learning of imbalanced data streams with concept drift. In *IJCAI*, pages 2393–2399, 2017.

[Lu et al., 2019] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang. Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12):2346–2363, 2019.

[Montiel et al., 2018] Jacob Montiel, Jesse Read, Albert Bifet, and Talel Abdessalem. Scikit-multiflow: A multi-output streaming framework. *The Journal of Machine Learning Research*, 19(1):2915–2914, 2018.

[Pedregosa et al., 2011] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. *the Journal of machine Learning research*, 12:2825–2830, 2011.

[Pratama and Wang, 2019] Mahardhika Pratama and Dian-hui Wang. Deep stacked stochastic configuration networks for lifelong learning of non-stationary data streams. *Information Sciences*, 495:150–174, 2019.

[Shao et al., 2018] J. Shao, F. Huang, Q. Yang, and G. Luo. Robust prototype-based learning on data streams. *IEEE Transactions on Knowledge and Data Engineering*, 30(5):978–991, 2018.

[Snell et al., 2017] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In *Advances in neural information processing systems*, pages 4077–4087, 2017.

[Yu et al., 2020] H. Yu, J. Lu, and G. Zhang. Online topology learning by a gaussian membership-based self-organizing incremental neural network. *IEEE Transactions on Neural Networks and Learning Systems*, 31(10):3947–3961, 2020.