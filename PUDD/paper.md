## Page 1

The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)

# Early Concept Drift Detection via Prediction Uncertainty

**Pengqian Lu**, **Jie Lu\***, **Anjin Liu**, **Guangquan Zhang**

Australian Artificial Intelligence Institute (AAII),
University of Technology Sydney, Ultimo, NSW 2007, Australia
{Pengqian.Lu@student., Jie.Lu@, Anjin.Liu@, Guangquan.Zhang@}uts.edu.au

## Abstract

Concept drift, characterized by unpredictable changes in data distribution over time, poses significant challenges to machine learning models in streaming data scenarios. Although error rate-based concept drift detectors are widely used, they often fail to identify drift in the early stages when the data distribution changes but error rates remain constant. This paper introduces the Prediction Uncertainty Index (PU-index), derived from the prediction uncertainty of the classifier, as a superior alternative to the error rate for drift detection. Our theoretical analysis demonstrates that: (1) The PU-index can detect drift even when error rates remain stable. (2) Any change in the error rate will lead to a corresponding change in the PU-index. These properties make the PU-index a more sensitive and robust indicator for drift detection compared to existing methods. We also propose a PU-index-based Drift Detector (PUDD) that employs a novel Adaptive PU-index Bucketing algorithm for detecting drift. Empirical evaluations on both synthetic and real-world datasets demonstrate PUDD’s efficacy in detecting drift in structured and image data.

**Code** — https://github.com/RocStone/PUDD
**Extended version** — https://arxiv.org/abs/2412.11158

## Introduction

In real-world applications, such as medical triage (Huggard et al. 2020) or time series forecasting tasks (Miyaguchi and Kajino 2019), the distribution of data may unpredictably change over time. This phenomenon, termed concept drift (Yuan et al. 2022), significantly degrades model performance. Moreover, drift also manifests between clients and servers in federated learning tasks (Jiang, Wang, and Dou 2022), or decision making process (Lu et al. 2020) further complicating the learning process. Error rate-based drift detection is one of the most popular approaches to handling concept drift due to its efficiency (Lu et al. 2018a). It continuously monitors the classifier’s error rate, issuing an alarm when this rate exceeds a preset threshold (Raab, Heusinger, and Schleif 2020; Frias-Blanco et al. 2014).

However, in the early stages of concept drift, a model’s error rate may remain stable. In such a case, error rate-based drift signals are unable to indicate the changes in data distribution. To address this, we explore alternative methods to detect distribution changes before a drop in error rate occurs. Upon rethinking the logic behind error rate-based drift signals, we realize that the distribution of prediction probability from a given model will change prior to the error rate itself. In addition, if the error rate does change, then the distribution of prediction probability must change too.

Prediction probability is just an example of a quantitative measure of predictive uncertainty in models. We believe that by clearly defining predictive uncertainty and demonstrating that it is a superior alternative to error rates, we can significantly enhance the sensitivity and robustness of drift detection. Therefore, we introduce the Prediction Uncertainty Index (PU-index), which measures the probability assigned by a classifier that an instance does not belong to the true class. Fig. 1 shows an illustrative example.

The objective of this paper is to address two key questions: (1) Can we identify a more effective drift detection signal that captures changes in data distribution when the error rate remains stable? (2) Can we theoretically prove that if the drift detection signal shows no significant change, then the model’s error rate will also remain stable? In other words, we seek to develop a superior drift detection signal that can detect drift when the error rate cannot, and to prove that if the new signal fails to detect drift, the error rate will also fail to do so. We believe that these criteria are essential for evaluating and comparing different drift detection signals.

The theoretical results provided in this paper show strong evidence for the superiority of the PU-index as a metric for concept drift detection. It exhibits at least equivalent sensitivity to error-based metrics and potentially higher sensitivity in certain scenarios, rendering it a more robust and comprehensive measure for identifying concept drift.

However, it could be argued that a more sensitive detection method might result in a higher false alarm rate, potentially overreacting to minor fluctuations in model performance. If there is no significant drift in the error rate, why should we still seek to detect it? To mitigate such concern, we employ the Chi-square test, a robust statistical significance test, to the PU-index for drift detection. This approach helps distinguish between meaningful distributional shifts and inconsequential variations, thereby maintaining the benefits of increased sensitivity while minimizing false alarms. The p-value obtained

\*Corresponding author.
Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

&lt;page_number&gt;19124&lt;/page_number&gt;

---


## Page 2

Denote test set 1 and 2 as:
D₁ = {dᵢ | i ∈ [0, t₁]}
D₂ = {dᵢ | i ∈ (t₁, t₂)}

Drift occurrence at time t₁.

Error rate remain stable at time t₂.

&lt;img&gt;A graph showing Error rate on the y-axis and Time on the x-axis. Two lines are plotted: one for Test Set 1 (blue) and one for Test Set 2 (red). The blue line remains relatively flat, indicating stability. The red line shows a sharp increase after t₁, indicating drift.&lt;/img&gt;

Data distribution of test set 1

&lt;img&gt;A scatter plot showing data points. Points are colored by class (Class 0 and Class 1), with some points highlighted around decision boundaries, indicating a distribution gap.&lt;/img&gt;
Early stage of concept drift.

Build histograms for prediction error on test set 1 and 2

&lt;img&gt;A histogram comparing Prediction Error for Test Set 1 (blue) and Test Set 2 (red). The histograms appear identical, suggesting no difference in prediction error.&lt;/img&gt;
Identical histograms for test set 1 and 2.

Data distribution of test set 2

Build histograms for PU-index on test set 1 and 2.

&lt;img&gt;A histogram comparing Prediction Uncertainty for Test Set 1 (blue) and Test Set 2 (red). The histograms differ, indicating a change in prediction uncertainty.&lt;/img&gt;
Different histograms for test set 1 and 2.

Figure 1: Illustrative example of the early stage of concept drift when an error rate-based detector fails to detect concept drift, but a prediction uncertainty-based detector can. The data around decision boundaries have been highlighted in the middle of the figure, showing the distribution gap between test sets 1 and 2. Such a gap implies concept drift occurrence. However, in this case, the error rates of the two test sets are the same. We also provide a theoretical proof in the Appendix to demonstrate the existence of such a case. By contrast, the distribution of prediction uncertainty has changed. The example implies that a prediction uncertainty-based detector can detect drift when an error rate-based detector fails.

from the Pearson’s Chi-square test serves as a precise control mechanism for our tolerance to false alarms. By adjusting the significance level (α), we can directly modulate the trade-off between sensitivity and false positive rate.

In this paper, we introduce PUDD, a drift detector that uses the Prediction Uncertainty (PU) index to identify concept drift. PUDD employs a sliding window approach to remove outdated data and split the historical stream into two samples. It then applies an Adaptive PU-index Bucketing algorithm to automatically construct histograms that meet our theoretical conditions. Using these histograms, we apply Pearson’s Chi-square test to determine if drift has occurred. Our main contributions are:

1. To the best of our knowledge, this is the first systematic study that compares two different drift detection signals with theoretical analysis.
2. We propose a novel drift detection metric called the PU-index, which is theoretically proven always to outperform error rate-based drift measurements. This provides crucial insight into the PU-index as a more sensitive and robust alternative for concept drift detection.
3. To identify concept drift in streaming data through the PU-index, we propose a Prediction Uncertainty index base drift detector. It comprises an Adaptive PU-index Bucketing algorithm to build a histogram for the PU-index, which meets the condition of our theoretical analysis, to conduct the Pearson’s Chi-square test and detect drift.

methods. The former directly addresses the root cause of drift—changes in the data distribution—while the latter focuses on variations in the model’s performance, often achieving higher computational efficiency.

## Data Distribution-based Methods

Data distribution-based methods measure shifts in the underlying distribution. For instance, a statistical density estimation approach is proposed in (Song et al. 2007), enabling the quantification of differences between two samples. Histogram-based techniques frequently serve to represent distributions in high-dimensional feature spaces (Liu et al. 2017). For example, (Boracchi et al. 2018) and (Yonekawa, Saito, and Kurokawa 2022) introduce hierarchical and dynamically adjustable strategies to construct histograms, respectively. Interval formation can also rely on methods like QuadTree (Coelho, Torres, and de Castro 2023) and K-means clustering (Liu, Lu, and Zhang 2020). Beyond direct histogram or density estimation, some methods incorporate contextual factors (Lu et al. 2018b) or anticipate future distributions. For example, (Cobb and Van Looveren 2022) uses a context-based CoDiTE function (Park et al. 2021) to detect drift, while (Li et al. 2022) exploits a predictive model for future distributions. There are also approaches that leverage Graph Neural Networks to track and adapt to distribution changes directly (Zhou et al. 2023). Although effective, these strategies can be computationally expensive in high-dimensional data streams (Souza et al. 2021). To date, no existing work uses histograms constructed from prediction uncertainty for drift detection, leaving a notable gap in the literature. To the best of our knowledge, there is no previous work that proposes building a histogram of prediction uncertainty to detect concept drift.

## Literature Review

In this section, we examine two approaches to concept drift detection: data distribution-based and error rate-based

&lt;page_number&gt;19125&lt;/page_number&gt;

---


## Page 3

Error Rate-based Methods

Error rate-based detectors are well-studied and computationally efficient. Approaches such as (Gama et al. 2004), (Baena-Garcia et al. 2006), and (Frias-Blanco et al. 2014) monitor variations in model error rates to detect drift. Adaptive window resizing is explored in (Bifet and Gavalda 2007), and forgetting mechanisms are introduced in (Jiao et al. 2022) to weight classifiers dynamically. More recent strategies apply Gaussian Mixture Models to compare windows (Yu et al. 2024) or enter reactive states upon detecting alarms (Tahmasbi et al. 2021). Despite their efficiency, error rate-based detectors struggle to identify drift when accuracy remains stable, particularly during its early stages (see Fig. 1). To address this limitation, we propose a prediction uncertainty-based approach, capable of detecting shifts even before error rates degrade. This method enhances early detection capabilities and complements existing drift detection strategies.

Preliminaries

Pearson’s Chi-Square Test

The Pearson’s Chi-square test assesses whether two categorical variables are independent. Its null hypothesis assumes independence, and if the computed p-value falls below a chosen significance level, the null hypothesis is rejected, indicating potential dependence between the variables.

The test’s reliability depends on having sufficiently large observed and expected frequencies. As noted in (EP 1978), the Chi-square test produces valid results when each observed count exceeds 50 and every expected count exceeds 5. Under these conditions, the distribution of the test statistic closely approximates a normal distribution, enhancing the test’s validity. The test relies on a contingency table. For the cell in the i-th row and j-th column, O<sub>ij</sub> denotes the observed frequency, and its expected frequency is given by:

E<sub>ij</sub> = n<sub>i</sub> × n<sub>j</sub>/N, (1)

where n<sub>i</sub> and n<sub>j</sub> are the cumulative frequencies of the respective row and column, and N is the sum of the table. The Chi-square test statistic is derived as:

χ² = ΣΣ<sub>i</sub><sub>j</sub> (O²<sub>ij</sub>/E<sub>ij</sub>) – N. (2)

Correspondingly, the p-value associated with χ² is:

p = 1 – ∫₀χ² x^(w/2 – 1) · e^(-x/2) dx/2^(w/2) · Γ(w/2), (3)

where w indicates the degrees of freedom, calculated as:

w = (number of columns – 1) × (number of rows – 1). (4)

Space Partitioning Algorithms

Space partitioning algorithms are widely studied to build histograms to establish density estimators (Silverman 2018). The key is to split feature space into partitions and count the instances falling into it to build a histogram. The partitions can be built by QuanTree (Boracchi et al. 2018), Kernel QuantTree (Stucchi et al. 2023), or Neural Network (Yonekawa, Saito, and Kurokawa 2022). Particularly, we introduce the Ei-kMeans space partitioning algorithm (Liu, Lu, and Zhang 2020), which can automatically determine the number and size of partitions.

Given two samples A and B, Ei-kMeans initializes centroids by iteratively selecting N/K points from a copy of A, where N = |A| and K is the hyperparameter of the kMeans algorithm. Each iteration: (1) Select the point in A with the largest 1-NN distance as z<sub>i</sub>. (2) Removes z<sub>i</sub> and its N/K-nearest neighbors from A. This process repeats N/K times, yielding N/K initial centroids. Then the kMeans algorithm is applied to A with the initial centroids to derive K clusters denoted as {C<sub>i</sub>|i ∈ [1, N/K]}.

To ensure that the number of examples in each cluster is larger than 5 to be able to conduct the Chi-square test, an amplify-shrink algorithm is proposed to adjust the number of examples in each cluster. Let us denote the number of instances in the clusters as V = {C<sub>i</sub>|i ∈ [1, N/K]}. The distance matrix between the examples in A and the centers of each cluster is denoted as M<sub>dist</sub> ∈ ℝ<sup>N×K</sup>, which is amplified by: M<sub>dist</sub> = M<sub>dist</sub> ⊙ (1 · e^(θ · (V/N – 1))), where 1 ∈ ℝ<sup>N×1</sup> is an all-ones matrix, θ denotes the hyperparameter controlling the shape of the coefficient function, . Let M<sub>dist</sub><sup>ij</sup> denote the amplified distance between i-th data A<sub>i</sub> and j-th center c<sub>j</sub>, the assigned cluster for A<sub>i</sub> is defined as y<sub>i</sub> = arg min<sub>j=1…K</sub> M<sub>dist</sub><sup>ij</sup>.

After the amplify-shrink algorithm, the final clusters are derived and can be considered as subspaces in the feature space of A. The numbers of examples of A and B in the clusters are counted to form a histogram.

Methodology

This section sets up the problem and provides a theoretical analysis demonstrating the advantages of the PU-index over error rate-based methods. We then introduce a novel sliding window strategy and an Adaptive PU-index Bucketing algorithm for concept drift detection and adaptation.

Problem Setup

Formally, we represent the streaming data collected during the period [1, t] as D<sub>1,t</sub> = {(x<sub>j</sub>, y<sub>j</sub>)|j ∈ [1, t]}. If the data is collected in chunks, then the stream includes a set of chunks D<sub>1,t</sub> = {D̄<sub>j</sub>|j ∈ [1, t]}, where each chunk D̄<sub>j</sub> = {(x<sub>jk</sub>, y<sub>jk</sub>)|k ∈ [1, M]} includes M examples. Here, x<sub>jk</sub> represents an instance with d dimensional attributes, y<sub>jk</sub> denotes the corresponding label, and M denotes the chunk size. In this paper, we focus only on the data collected in chunks. If the stream D<sub>1,t</sub> follows a distribution P<sub>1,t</sub>(x, y), following (Lu et al. 2018a), we claim that a drift occurs at time t + 1 if

P<sub>1,t</sub>(x, y) ≠ P<sub>t,∞</sub>(x, y). (5)

The goal of concept drift detection is to raise an alarm at time t + 1 when the distribution of data changes. The most popular metric for detecting the distribution change is prediction error. For a classifier f, assuming the number of

&lt;page_number&gt;19126&lt;/page_number&gt;

---


## Page 4

mermaid
flowchart TD
    A[Step 1. Build sliding windows] -->|A stream| D{Antiquated Data}
    A -->|PU-index Bank| E[Build window pairs by cut point exploration.]
    E --> F[Classifier f]
    E --> G[u<sub>t<sub>1</sub></u>]
    F --> H["pu<sub>t<sub>n+1</sub></sub>"]
    D -->|If last drift detected at t<sub>1</sub>: discard antiquated data.| I
    I --> D
    I --> G

    J[Step 2. Build bins] -->|a) For each window pair Cutting point| K[W<sub>1</sub>]
    K --> L[pu<sub>1</sub>]
    K --> M[W<sub>2</sub>]
    M --> N[pu<sub>n</sub>]
    M --> O[pu<sub>n+1</sub>]

    J -->|b) Build bins by PU-index bucketing algorithm| P[0,?,?], Q[?,?,?]..., R[?,1]]
    P --> S[t<sub>1</sub>]
    Q --> T[t<sub>1</sub>]

    J -->|Step 3. Drift detection with chi-square test| U[a) Compute the contingency table by counting the instances falling into bins in each window pair.]

    U --> V[n<sup>(1)</sup><sub>b<sub>1</sub></sub> n<sup>(1)</sup><sub>b<sub>2</sub></sub> ... ..., ..., ...]
    U --> W[n<sup>(2)</sup><sub>b<sub>1</sub></sub> n<sup>(2)</sup><sub>b<sub>2</sub></sub> ... ..., ..., ...]

    U -->|b) Compute the p-value| X[p = 1 - ∫₀<sup>x²</sup><sub>w<sup>(w-1)</sup> / (2²) . e<sup>-x/2</sup> dx]

    X --> Y[c) Condition of raise drift alarm]
    Y -->|If p ≤ σ: Raise drift detected alarm.| Z[Else if all window pairs examined: No drift detected.]
    Z --> A1[Else: Go to Step 2.]
```

Figure 2: The framework of our proposed algorithm. The sliding window strategy has two components, i.e., antiquated data discard and cutting point exploration as shown on the left. The Adaptive PU-index Bucketing algorithm is shown in the middle. The drift detection process is shown on the right.

classes is *n*, the classifier will output a prediction probability for each class $y \in \mathbb{R}^{n}$ and $\sum_{i=1}^{n} \hat{y} = 1$. The prediction error of an instance $x_i$ is defined as:

$$e_i = \mathbb{I}(\hat{y}_j = \arg\max_j f_j(x_i) \neq y_i),$$

(6)

where $\mathbb{I}(\cdot)$ is the indicator function.

As we mentioned earlier, our motivation is that the prediction probability will intuitively change before the prediction error when drift occurs. In this paper, we measure the prediction probability by the PU-index which is defined as:

$$u_i = 1 - f_{y_i}(x_i),$$

(7)

where $f_{y_i}(x_i)$ denotes the probability predicted by the classifier that $x_i$ belongs to the ground truth class $y_i$.

**Theoretical Analysis of Error Rate and PU-index**

To rigorously evaluate the efficacy of these two metrics for concept drift detection, we conduct a theoretical comparison from two complementary perspectives. (1) When the PU-index distribution remains stable, potentially failing to detect concept drift, we investigate whether the error rate distribution exhibits changes that could indicate drift. (2) Conversely, when the error rate distribution remains constant, we examine whether the PU-index distribution demonstrates changes that might reveal underlying changes in the data stream.

**Theorem 1.** Let $W_1$ and $W_2$ be two windows of a data stream in a multi-class classification problem. If their respective PU-index histograms $H_1$ and $H_2$ are identical, where the histograms are constructed such that the first bin contains all misclassified instances and the remaining bins partition the misclassified instances, then the error rates and error standard deviations of $W_1$ and $W_2$ are equal.

**Theorem 2.** Given a multi-class classification problem, if two windows have equal error standard deviations or error rates, their PU-index histograms, where the first bin contains all correctly classified instances and the remaining bins partition the misclassified instances, may not have identical bin proportions.

Due to the page limit, the proofs are provided in the Appendix. These theorems lead to the following conclusions: (1) Theorem 1 demonstrates that when the PU-index distribution remains stable, the error rate and the error standard deviation also remain constant. This implies that if the PU-index fails to detect concept drift, error-based metrics will also fail to detect it. (2) Theorem 2 establishes that even when error rates and error standard deviations are equal between two windows, the PU-index distributions may differ. This suggests that the PU-index has the potential to detect subtle changes in the data distribution that are not captured by traditional error-based metrics. These findings show that the PU-index offers at least the same sensitivity as error-based metrics and potentially higher sensitivity in certain scenarios, making it a more robust and comprehensive measure for detecting concept drift in streaming data environments.

**Sliding Window Strategy and Adaptive PU-index Bucketing Algorithm**

To detect concept drift using the PU-index without making it overly sensitive, we apply the Chi-square test to examine the PU-index distribution. The hypotheses are:

**Null Hypothesis ($H_0$):** The PU-index distribution does not change over time, indicating no concept drift.

**Alternative Hypothesis ($H_1$):** The PU-index distribution changes over time, indicating concept drift.

If the Chi-square statistic exceeds the critical value at the chosen significance level, we reject $H_0$ and conclude that

&lt;page_number&gt;19127&lt;/page_number&gt;

---


## Page 5

<table>
<tr>
<td></td>
<td><strong>Plane</strong></td>
<td><strong>Car</strong></td>
<td><strong>Bird</strong></td>
<td><strong>Cat</strong></td>
<td>...</td>
<td><strong>Truck</strong></td>
</tr>
<tr>
<td><strong>Plane</strong></td>
<td>0.98</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
<td>...</td>
<td>0.01</td>
</tr>
<tr>
<td><strong>Car</strong></td>
<td>0.01</td>
<td>0.89</td>
<td>0.01</td>
<td>0.01</td>
<td>...</td>
<td>0.01</td>
</tr>
<tr>
<td><strong>Bird</strong></td>
<td>0.01</td>
<td>0.01</td>
<td>0.91</td>
<td>0.01</td>
<td>...</td>
<td>0.01</td>
</tr>
<tr>
<td><strong>Cat</strong></td>
<td>0.00</td>
<td>0.02</td>
<td>0.01</td>
<td>0.92</td>
<td>...</td>
<td>0.01</td>
</tr>
<tr>
<td>⋮</td>
<td>⋮</td>
<td>⋮</td>
<td>⋮</td>
<td>⋮</td>
<td>...</td>
<td>⋮</td>
</tr>
<tr>
<td><strong>Truck</strong></td>
<td>0.00</td>
<td>0.01</td>
<td>0.01</td>
<td>0.01</td>
<td>...</td>
<td>0.92</td>
</tr>
</table>

Table 1: Transition matrix of generating CIFAR-10-CD through the Markov process. The classes marked in bold represent the user’s initial interest and are considered positive labels. All other classes are considered negative labels.

concept drift has occurred. Otherwise, we detect no significant drift. Thus, detecting concept drift using Chi-square involves two key steps: (1) partitioning the data stream into two windows and (2) constructing a histogram of the collected PU-index values.

We adopt a sliding window strategy to handle online streaming data. Let $D_{1,t} = \{\bar{D}_j | j \in [1,t]\}$ denote PU-index chunks collected from the start of the stream, potentially from different distributions. For instance, suppose $D_{1,t_1}$ and $D_{t_1,t}$ differ in distribution. If a new chunk $\bar{D}_{t+1}$ matches the distribution of $D_{t_1,t}$, no drift should be detected. However, keeping antiquated data $D_{1,t_1}$ could trigger a false alarm, since $D_{1,t_1}$ and $D_{t_1,t+1}$ differ in distribution.

To solve this “antiquated distribution” problem, we discard outdated data after detecting a drift at time $t_1$. Subsequent drift detection uses only $D_{t_1,t+1}$, avoiding false alarms caused by old data. After discarding antiquated data, we must determine how to form two windows on the current substream. We do this by exploring all possible cutting points $r \in [t_1,t+1]$. Thus, $D_{t_1,t+1}$ is split into $D_{t_1,r}$ and $D_{r,t+1}$ for the Adaptive PU-index Bucketing algorithm. The sliding window is illustrated on the left side of Fig. 2.

Theoretical analysis shows that misclassified instances’ PU-indices must be grouped into the same bin. For counterpart, we use Ei-kMeans to form bins that meet the Chi-square test requirements. We call this the Adaptive PU-index Bucketing algorithm, illustrated in the middle of Fig. 2.

## PU-Index based Drift Detector

In this subsection, we introduce the overall of our method. Firstly, given a substream containing the recent chunks’ PU-index $u_{t_1,t}$, we explore all cutting points $r \in [t_1,t]$. Based on the cutting points, we have $t - t_1$ window pairs, denoted as $\boldsymbol{u}_{t_1,r}$, and $\boldsymbol{u}_{r,t}$. Then we defined the PU-index pairs for correctly and wrongly classified instances as:

$$\boldsymbol{u}_{t_1,r}^C = \{u | u_i \in \boldsymbol{u}_{t_1,r} \land \hat{y}_i = y_i\}, \quad (8)$$
$$\boldsymbol{u}_{r,t}^C = \{u | u_i \in \boldsymbol{u}_{r,t} \land \hat{y}_i = y_i\}, \quad (9)$$
$$\boldsymbol{u}_{t_1,r}^M = \{u | u_i \in \boldsymbol{u}_{t_1,r} \land \hat{y}_i \neq y_i\}, \quad (10)$$
$$\boldsymbol{u}_{r,t}^M = \{u | u_i \in \boldsymbol{u}_{r,t} \land \hat{y}_i \neq y_i\}. \quad (11)$$

These four equations represent the PU-index for correctly and misclassified instances in the first and second windows, respectively. Therefore, $\boldsymbol{u}_{t_1,r} = \{\boldsymbol{u}_{t_1,r}^C, \boldsymbol{u}_{t_1,r}^M\}$ and $\boldsymbol{u}_{r,t} = \{\boldsymbol{u}_{r,t}^C, \boldsymbol{u}_{r,t}^M\}$. Next, we compute the contingency table $T \in \mathbb{R}^{2 \times (K+1)}$, where $K$ is a hyperparameter in Ei-kMeans. To calculate $T$, we apply the Adaptive PU-index Bucketing algorithm on $\boldsymbol{u}_{t_1,r}^C$ to build a histogram. Then we count the instances of $\boldsymbol{u}_{t_1,r}^C$ that fall into the histogram bins and fill them in $T_{1i}$ where $i \in [1,K]$. Likewise, we count the examples of $\boldsymbol{u}_{r,t}^C$ falling into the previously obtained bins and fill them in $T_{2i}$. Finally, we fill in $T_{1,K+1}$ and $T_{2,K+1}$ with the size of $\boldsymbol{u}_{t_1,r}^M$ and $\boldsymbol{u}_{r,t}^M$. The expected frequency of $T_{ij}$ is defined as:

$$E_{ij} = \frac{\sum_{j=1}^{K+1} T_{ij} \times \sum_{i=1}^{2} T_{ij}}{\sum_{ij} T_{ij}}. \quad (12)$$

The Chi-square test statistic is defined as:

$$\chi^2 = \sum_{i} \sum_{j} \left(\frac{T_{ij}^2}{E_{ij}}\right) - \sum_{ij} T_{ij}. \quad (13)$$

Finally, the p-value is computed by:

$$p = 1 - \int_0^{\chi^2} \frac{x^{\frac{K}{2}-1} \cdot e^{-\frac{x}{2}}}{2^{\frac{K}{2}} \cdot \Gamma\left(\frac{K}{2}\right)} dx. \quad (14)$$

Based on the Equation (12-14), we can compute the p-value for each window pair $D_{t_1,r}$, and $D_{r,t}$. If the minimum p-value among all window pairs is smaller than a predefined threshold $\sigma$, we raise a drift detected alarm. It is important to clarify that the specified threshold controls the Type I error rate for each individual window pair test rather than the overall Type I Error across the entire substream. Consequently, we do not employ multiple comparison adjustments since our statistical guarantees apply at the single-test level rather than the family-wise level. The pseudo-code and time complexity analysis is provided in the Appendix.

## Experiments

In this section, we introduce the settings and results of the experiments in our paper. The details of implementation, datasets, baselines, and the critical difference diagrams (Ismail Fawaz et al. 2019) for the experiments in this paper are introduced in the Appendix.

## Datasets and Baselines

We propose CIFAR-10-CD, a synthetic concept drift image dataset with transition matrix shown in Table 1, to simulate user interests changing via a Markov process. Initially, three CIFAR-10 classes are marked positive, with interest shifts occurring probabilistically (e.g. 1% chance of Plane to Horse transfer). Our experiments utilize 3 real-world datasets (airline(Ikonomovska 2011), elec2(Harries 1999), powersupply(Dau et al. 2019)) and 4 synthetic sets (sine(Gama et al. 2004), mixed(Gama et al. 2004), CIFAR-10-CD, sea variants(Bifet et al. 2010)). We compare against 7 classic detectors (ADWIN(Bifet and Gavalda 2007), DDM(Gama et al. 2004), EDDM(Baena-Garcia et al. 2006), HDDM-A(Frias-Blanco et al. 2014), HDDM-W(Frias-Blanco et al. 2014), KSWIN(Raab, Heusinger, and Schleif 2020), PH(Sebastião

&lt;page_number&gt;19128&lt;/page_number&gt;

---


## Page 6

<table>
<thead>
<tr>
<th></th>
<th></th>
<th colspan="6">Incremental Training</th>
<th colspan="6">Training only at Initialization or Adaptation</th>
</tr>
<tr>
<th>Classifier</th>
<th>ddm name</th>
<th>airline-I</th>
<th>elec2-I</th>
<th>mixed-I</th>
<th>ps-I</th>
<th>sea0-I</th>
<th>sine-I</th>
<th>airline-O</th>
<th>elec2-O</th>
<th>mixed-O</th>
<th>ps-O</th>
<th>sea0-O</th>
<th>sine-O</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">DNN</td>
<td>ADWIN</td>
<td>61.65</td>
<td>71.94</td>
<td>78.45</td>
<td>71.12</td>
<td>97.70</td>
<td>79.27</td>
<td>59.36</td>
<td>69.71</td>
<td>84.41</td>
<td>65.95</td>
<td>95.18</td>
<td>86.59</td>
</tr>
<tr>
<td>DDM</td>
<td>61.29</td>
<td>71.02</td>
<td><u>80.72</u></td>
<td>71.37</td>
<td>96.86</td>
<td><u>86.41</u></td>
<td>56.79</td>
<td>69.51</td>
<td>83.03</td>
<td>69.35</td>
<td>93.55</td>
<td>80.32</td>
</tr>
<tr>
<td>EDDM</td>
<td>62.13</td>
<td>70.39</td>
<td>78.01</td>
<td>65.21</td>
<td>96.89</td>
<td><u>75.34</u></td>
<td><u>61.17</u></td>
<td>69.22</td>
<td>71.16</td>
<td>67.57</td>
<td>92.18</td>
<td>76.54</td>
</tr>
<tr>
<td>HDDM-A</td>
<td>62.70</td>
<td>71.16</td>
<td>76.70</td>
<td>68.97</td>
<td>97.84</td>
<td>81.65</td>
<td>59.13</td>
<td>68.98</td>
<td>84.27</td>
<td>67.81</td>
<td>95.11</td>
<td><u>86.82</u></td>
</tr>
<tr>
<td>HDDM-W</td>
<td>61.50</td>
<td>71.23</td>
<td>77.07</td>
<td>68.74</td>
<td>97.84</td>
<td><u>85.43</u></td>
<td><u>62.42</u></td>
<td>68.48</td>
<td>84.32</td>
<td>66.89</td>
<td>92.24</td>
<td>86.57</td>
</tr>
<tr>
<td>KSWIN</td>
<td>63.02</td>
<td>70.56</td>
<td>78.58</td>
<td>70.80</td>
<td>97.88</td>
<td>79.00</td>
<td><u>61.34</u></td>
<td>69.08</td>
<td><u>84.43</u></td>
<td>66.46</td>
<td>91.60</td>
<td>83.66</td>
</tr>
<tr>
<td>PH</td>
<td>62.15</td>
<td>72.25</td>
<td>79.49</td>
<td>70.86</td>
<td>97.73</td>
<td>78.07</td>
<td>60.36</td>
<td>68.24</td>
<td>84.36</td>
<td>68.83</td>
<td><u>95.21</u></td>
<td><u>86.88</u></td>
</tr>
<tr>
<td rowspan="3"></td>
<td>PUDD-1</td>
<td><u>63.31</u></td>
<td><u>74.92</u></td>
<td>77.39</td>
<td><u>72.25</u></td>
<td><u>97.94</u></td>
<td><u>86.19</u></td>
<td>60.90</td>
<td>69.35</td>
<td>82.65</td>
<td><u>71.47</u></td>
<td>94.89</td>
<td>83.39</td>
</tr>
<tr>
<td>PUDD-3</td>
<td><u>63.21</u></td>
<td><u>74.93</u></td>
<td><u>80.05</u></td>
<td><u>72.23</u></td>
<td><u>98.04</u></td>
<td>85.12</td>
<td>60.16</td>
<td>68.98</td>
<td><u>84.65</u></td>
<td><u>70.37</u></td>
<td><u>95.99</u></td>
<td>84.97</td>
</tr>
<tr>
<td>PUDD-5</td>
<td><u>63.35</u></td>
<td><u>74.92</u></td>
<td><u>82.81</u></td>
<td><u>72.24</u></td>
<td><u>98.23</u></td>
<td>82.51</td>
<td>60.19</td>
<td>68.68</td>
<td><u>84.90</u></td>
<td><u>70.20</u></td>
<td><u>96.29</u></td>
<td>85.09</td>
</tr>
<tr>
<td rowspan="7">GNB</td>
<td>ADWIN</td>
<td>50.17</td>
<td>68.90</td>
<td><u>83.95</u></td>
<td>70.06</td>
<td>94.18</td>
<td>82.49</td>
<td><u>54.66</u></td>
<td><u>68.30</u></td>
<td>83.62</td>
<td>68.78</td>
<td>93.93</td>
<td>82.45</td>
</tr>
<tr>
<td>DDM</td>
<td>52.94</td>
<td>67.75</td>
<td>83.82</td>
<td>69.63</td>
<td>93.97</td>
<td>82.07</td>
<td>52.43</td>
<td>67.60</td>
<td><u>83.59</u></td>
<td>67.52</td>
<td>93.48</td>
<td>81.90</td>
</tr>
<tr>
<td>EDDM</td>
<td><u>62.72</u></td>
<td>67.73</td>
<td>83.19</td>
<td>70.04</td>
<td>94.06</td>
<td>83.12</td>
<td>54.11</td>
<td>67.64</td>
<td>74.65</td>
<td><u>70.04</u></td>
<td><u>94.06</u></td>
<td>73.88</td>
</tr>
<tr>
<td>HDDM-A</td>
<td>52.80</td>
<td>67.73</td>
<td>83.92</td>
<td>70.87</td>
<td>94.28</td>
<td>83.25</td>
<td><u>55.62</u></td>
<td><u>67.73</u></td>
<td>83.66</td>
<td><u>71.24</u></td>
<td>93.96</td>
<td><u>83.36</u></td>
</tr>
<tr>
<td>HDDM-W</td>
<td>48.66</td>
<td>67.73</td>
<td>83.91</td>
<td>71.06</td>
<td>92.11</td>
<td>82.83</td>
<td>48.62</td>
<td>67.71</td>
<td>83.60</td>
<td>69.53</td>
<td>91.63</td>
<td>83.25</td>
</tr>
<tr>
<td>KSWIN</td>
<td>49.84</td>
<td>67.87</td>
<td>83.92</td>
<td>71.23</td>
<td>91.72</td>
<td>81.88</td>
<td>48.83</td>
<td>67.63</td>
<td>83.59</td>
<td>67.99</td>
<td>90.02</td>
<td>81.32</td>
</tr>
<tr>
<td>PH</td>
<td>49.35</td>
<td>70.12</td>
<td>83.88</td>
<td>70.36</td>
<td>94.34</td>
<td><u>83.54</u></td>
<td>49.02</td>
<td><u>70.04</u></td>
<td>83.61</td>
<td>68.67</td>
<td>94.12</td>
<td>83.10</td>
</tr>
<tr>
<td rowspan="3"></td>
<td>PUDD-1</td>
<td><u>53.57</u></td>
<td><u>70.85</u></td>
<td>82.99</td>
<td><u>71.88</u></td>
<td><u>94.61</u></td>
<td>83.12</td>
<td>51.05</td>
<td>62.76</td>
<td>79.23</td>
<td><u>71.13</u></td>
<td><u>94.25</u></td>
<td>81.48</td>
</tr>
<tr>
<td>PUDD-3</td>
<td><u>53.03</u></td>
<td><u>70.85</u></td>
<td><u>83.92</u></td>
<td><u>71.59</u></td>
<td><u>94.81</u></td>
<td><u>83.39</u></td>
<td>49.45</td>
<td>59.32</td>
<td>83.58</td>
<td><u>71.20</u></td>
<td><u>94.60</u></td>
<td>83.80</td>
</tr>
<tr>
<td>PUDD-5</td>
<td><u>52.16</u></td>
<td><u>70.69</u></td>
<td><u>84.12</u></td>
<td><u>71.59</u></td>
<td><u>94.85</u></td>
<td><u>83.43</u></td>
<td><u>54.37</u></td>
<td>59.44</td>
<td><u>83.96</u></td>
<td>70.40</td>
<td><u>94.62</u></td>
<td><u>83.38</u></td>
</tr>
<tr>
<td rowspan="7">VFDT</td>
<td>ADWIN</td>
<td>60.39</td>
<td><u>73.98</u></td>
<td>84.30</td>
<td>71.69</td>
<td>94.77</td>
<td>87.11</td>
<td>61.22</td>
<td><u>74.29</u></td>
<td>83.54</td>
<td>68.78</td>
<td>92.96</td>
<td><u>85.74</u></td>
</tr>
<tr>
<td>DDM</td>
<td>60.16</td>
<td><u>74.82</u></td>
<td>84.14</td>
<td>70.68</td>
<td>94.86</td>
<td>86.50</td>
<td>59.28</td>
<td><u>74.75</u></td>
<td>82.31</td>
<td>67.53</td>
<td>93.63</td>
<td>82.23</td>
</tr>
<tr>
<td>EDDM</td>
<td>61.19</td>
<td>73.81</td>
<td>83.15</td>
<td>70.06</td>
<td>94.17</td>
<td>85.59</td>
<td><u>62.31</u></td>
<td>73.81</td>
<td>73.73</td>
<td>70.06</td>
<td>93.60</td>
<td>77.72</td>
</tr>
<tr>
<td>HDDM-A</td>
<td>60.95</td>
<td>73.90</td>
<td><u>84.40</u></td>
<td>70.84</td>
<td><u>95.20</u></td>
<td><u>87.53</u></td>
<td>60.29</td>
<td>73.83</td>
<td><u>83.66</u></td>
<td><u>71.24</u></td>
<td>93.93</td>
<td>85.21</td>
</tr>
<tr>
<td>HDDM-W</td>
<td>61.11</td>
<td>73.80</td>
<td><u>84.40</u></td>
<td>70.99</td>
<td>93.50</td>
<td><u>87.45</u></td>
<td><u>61.92</u></td>
<td>73.73</td>
<td>83.59</td>
<td>69.54</td>
<td>91.76</td>
<td>85.28</td>
</tr>
<tr>
<td>KSWIN</td>
<td><u>61.30</u></td>
<td><u>74.10</u></td>
<td><u>84.42</u></td>
<td>71.27</td>
<td>93.40</td>
<td>86.22</td>
<td><u>62.06</u></td>
<td><u>74.10</u></td>
<td>83.57</td>
<td>67.51</td>
<td>89.84</td>
<td>82.60</td>
</tr>
<tr>
<td>PH</td>
<td>60.95</td>
<td>73.70</td>
<td>83.56</td>
<td>70.88</td>
<td>94.69</td>
<td>87.15</td>
<td>60.97</td>
<td>73.99</td>
<td>83.30</td>
<td>68.67</td>
<td>93.85</td>
<td>85.19</td>
</tr>
<tr>
<td rowspan="3"></td>
<td>PUDD-1</td>
<td>61.38</td>
<td>73.86</td>
<td>84.25</td>
<td><u>71.77</u></td>
<td>95.13</td>
<td>87.33</td>
<td>61.16</td>
<td>69.79</td>
<td>82.13</td>
<td><u>71.13</u></td>
<td><u>94.10</u></td>
<td>82.21</td>
</tr>
<tr>
<td>PUDD-3</td>
<td><u>61.57</u></td>
<td>73.84</td>
<td>83.94</td>
<td><u>71.79</u></td>
<td><u>95.21</u></td>
<td>87.42</td>
<td>59.90</td>
<td>69.79</td>
<td>84.04</td>
<td><u>71.20</u></td>
<td><u>94.63</u></td>
<td><u>85.81</u></td>
</tr>
<tr>
<td>PUDD-5</td>
<td><u>61.57</u></td>
<td>73.64</td>
<td>84.01</td>
<td><u>71.79</u></td>
<td><u>95.24</u></td>
<td><u>87.63</u></td>
<td>57.04</td>
<td>71.83</td>
<td><u>84.16</u></td>
<td>70.40</td>
<td><u>94.56</u></td>
<td><u>86.01</u></td>
</tr>
</tbody>
</table>

Table 2: Comparative analysis against classic drift detectors across 3 synthetic and 3 real-world datasets. The top 3 results are highlighted in bold and the top 1 results are in both bold and underlined. PUDD-x represents the threshold set as $10^{-x}$ for our method. The ps is short of powersupply dataset. Results for dataset sea10 and sea20 is provided in Appendix.

and Fernandes 2017)) and 5 SOTA methods (MCDD(Wan, Liang, and Yoon 2024), AMF(Mourtada, Gaiffas, and Scornet 2021), IWE(Jiao et al. 2022), NS(Wang et al. 2021), and ADLTER(Wang et al. 2022)).

## Comparison with Baselines and Ablation Studies

In this subsection, we compare our method with 7 classic drift detectors and 5 SOTA methods on 9 datasets (including a variant of the SEA dataset). Due to page constraints, results for SEA10 and SEA20 appear in the Appendix. We evaluate all methods using three classifiers—DNN (architecture detailed in the Appendix), Gaussian Naive Bayes (GNB) (Virtanen et al. 2020), and VFDT (Hulten, Spencer, and Domingos 2001)—under two training regimes: incremental (dataset-I) and one-time training at an alarm (dataset-O).

Results for the comparison with classic detectors are presented in Table 2, and those for SOTA methods are given in Table 3. For CIFAR-10-CD, due to its learning complexity, we only use incremental training and report results in Fig. 3. Our method is denoted as PUDD-X, where X represents the exponent in $10^{-X}$. Based on these experiments, we derive 6 observations. We introduce 4 of them here and leave the remaining 2 in the Appendix.

**Observation 1:** our method shows stronger performance compared to classic drift detectors as evidenced by the results presented in Table 2, Fig. 3, and additional results in Appendix. In incremental learning settings, PUDD ranks first in 17 out of 24 cases across different datasets and classifiers, and it is in the top 3 in 20 out of 24 cases. When trained only initially or with adaptation, it still performs well. It achieves first rank in 15 cases and top 3 in 19 cases. This shows that PUDD is particularly effective with incremental training. Results in Fig. 3 show PUDD outperformed all the baselines, which demonstrates the superiority of our method in detecting the concept drift in the image dataset. The critical difference diagram of the experiment in the Appendix shows that the PUDD is statistically significantly outperforms SOTA methods.

**Observation 2:** PUDD performs better with a smaller threshold as revealed in Table 2 and additional results in Ap-

&lt;page_number&gt;19129&lt;/page_number&gt;

---


## Page 7

<table>
<thead>
<tr>
<th>ddm name</th>
<th>airline</th>
<th>elec2</th>
<th>mixed</th>
<th>ps</th>
<th>sea0</th>
<th>sine</th>
</tr>
</thead>
<tbody>
<tr>
<td>AMF</td>
<td>38.56</td>
<td>66.24</td>
<td>49.49</td>
<td>69.63</td>
<td>93.67</td>
<td>49.52</td>
</tr>
<tr>
<td>IWE</td>
<td>38.02</td>
<td>68.90</td>
<td>49.47</td>
<td>64.10</td>
<td>93.14</td>
<td>49.51</td>
</tr>
<tr>
<td>NS</td>
<td><u>67.91</u></td>
<td>76.42</td>
<td>81.09</td>
<td>72.39</td>
<td>93.54</td>
<td>91.01</td>
</tr>
<tr>
<td>ADLTER</td>
<td><u>70.00</u></td>
<td>76.10</td>
<td>87.63</td>
<td>72.48</td>
<td>93.40</td>
<td><u>92.18</u></td>
</tr>
<tr>
<td>MCD-DD</td>
<td>63.65</td>
<td>69.81</td>
<td>86.68</td>
<td>71.66</td>
<td>97.66</td>
<td>90.21</td>
</tr>
<tr>
<td>PUDD-1</td>
<td>63.78</td>
<td><u>77.28</u></td>
<td><u>89.51</u></td>
<td><u>72.68</u></td>
<td><u>98.47</u></td>
<td><u>94.52</u></td>
</tr>
<tr>
<td>PUDD-3</td>
<td><u>64.62</u></td>
<td>76.77</td>
<td><u>89.47</u></td>
<td><u>72.79</u></td>
<td><u>98.44</u></td>
<td><u>94.76</u></td>
</tr>
<tr>
<td>PUDD-5</td>
<td>64.45</td>
<td>76.92</td>
<td><u>89.37</u></td>
<td><u>72.74</u></td>
<td><u>98.49</u></td>
<td>90.90</td>
</tr>
</tbody>
</table>

Table 3: Comparison with SOTA methods. The dataset ps is short for powersupply. The results for dataset sea10 and sea20 is provided in Appendix. The table shows that our methods PUDD in achieved top-1 in 5 out of 6 datasets, implying the effectiveness of PUDD compared with SOTA methods.

Appendix. In incremental learning scenarios, PUDD-1, PUDD-3, and PUDD-5 achieve top 1 in 5, 5, and 8 cases respectively. When tested in training only once until alarm way, PUDD-1, PUDD-3, and PUDD-5 achieved top 1 in 2, 6, and 8 cases respectively. PUDD consistently shows improved performance at lower thresholds in both scenarios. As detailed in the Sensitivity of PU-index section, a drift alarm triggers when the p-value is below the threshold, with lower thresholds indicating stricter conditions for alarm detection. Therefore, PUDD’s better performance with smaller thresholds suggests a high sensitivity to drift.

**Observation 3:** PUDD shows very competitive performance compared to SOTA methods. As shown in Table 3 and additional results in Appendix, our method attains the top rank in 7 out of 8 cases. On certain datasets, this improvement is particularly pronounced. For instance, PUDD-5 achieves a 98.49% accuracy, which is 2.8% higher than the best SOTA method. The only exception occurs in the airline dataset, where NS and ADLTER outperform PUDD.

This discrepancy can be explained by the airline dataset’s tabular nature and its numerous attributes, which are more effectively modeled through tree-based ensemble learning utilized by these SOTA methods. Moreover, these methods adapt to drift by adjusting ensembles rather than discarding and retraining them. In contrast, PUDD relies on retraining classifiers solely on recent data, which may not be suitable for attribute-rich datasets like the airline dataset. Nevertheless, for all other datasets, the results confirm that PUDD surpasses SOTA methods, thereby underscoring its overall superiority.

**Observation 4:** The Adaptive PU-index Bucketing algorithm outperforms Ei-kMeans. Figure 4 shows that PUDD surpasses Ei-kMeans across various datasets, classifier training methods, and threshold settings. The critical difference diagram in the Appendix shows that improvements at thresholds $10^{-3}$ and $10^{-5}$ are statistically significant.

In summary, these results confirm the theoretical benefits of the PU-index for drift detection. PUDD outperforms both classic and SOTA detectors, and the Adaptive PU-index Bucketing algorithm shows significant improvements over Ei-kMeans. This validates the PU-index as a sensitive, robust indicator capable of detecting drift even when error rates remain unchanged, thereby overcoming a major shortcoming of error rate-based approaches.

&lt;img&gt;
A horizontal bar chart titled "Prediction Accuracy" comparing various drift detection methods on CIFAR-10-CD.
The x-axis ranges from 75.0 to 78.5.
The y-axis lists the methods:
HDDM-A
DDM
HDDM-W
ADWIN
EDDM
KSWIN
PH
MCDD
PUDD-1
PUDD-3
PUDD-5

The bars (from left to right) show the following approximate values:
HDDM-A: ~75.5
DDM: ~76.0
HDDM-W: ~76.5
ADWIN: ~76.5
EDDM: ~76.5
KSWIN: ~76.5
PH: ~77.0
MCDD: ~77.5
PUDD-1: ~77.5
PUDD-3: ~78.0
PUDD-5: ~78.0
&lt;/img&gt;

Figure 3: Comparison with baselines on CIFAR-10-CD, excluding methods unable to detect drift in image datasets.

&lt;img&gt;
A horizontal bar chart titled "Average Accuracy".
The x-axis ranges from 76 to 80.
The y-axis lists the methods:
GNB - 1
GNB - 3
GNB - 5
VFDT - 1
VFDT - 3
VFDT - 5
DNN - 1
DNN - 3
DNN - 5

The bars (from left to right) show the following approximate values:
GNB - 1: ~76.5
GNB - 3: ~76.0
GNB - 5: ~76.0
VFDT - 1: ~77.0
VFDT - 3: ~78.0
VFDT - 5: ~79.0
DNN - 1: ~78.0
DNN - 3: ~79.0
DNN - 5: ~80.0

Legend:
EK
PUDD
&lt;/img&gt;

Figure 4: Accuracy comparison between PUDD (using Adaptive PU-index Bucketing) and Ei-kMeans (EK). We show average accuracy across 9 datasets using 3 classifiers.

Conclusion and Future Work

In our study, we demonstrated that the PU-index, as opposed to the error rate, is a more effective measure for detecting concept drift in machine learning models. We utilized the Adaptive PU-index bucketing algorithm to partition the PU-index and the Chi-square test to detect concept drift. We also introduced a technique for inducing concept drift in image datasets by simulating changes in user interest. We validated our method through experiments on both synthetic and real-world datasets. Future work should focus on automating drift alarm threshold determination, as current methods rely on manual settings that may not remain optimal over time. Our research also uncovers a method for generating multi-stream concept drift in image datasets by emulating shifts in user interests using Markov matrices, offering valuable insights for research in multistream concept drift learning.

Acknowledgments

This work was supported by the Australian Research Council through the Laureate Fellow Project under Grant FL190100149.

&lt;page_number&gt;19130&lt;/page_number&gt;

---


## Page 8

# References

Baena-Garcia, M.; del Campo-Ávila, J.; Fidalgo, R.; Bifet, A.; Gavalda, R.; and Morales-Bueno, R. 2006. Early drift detection method. In *Fourth international workshop on knowledge discovery from data streams*, volume 6, 77–86. Citeseer.

Bifet, A.; and Gavalda, R. 2007. Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM international conference on data mining*, 443–448. SIAM.

Bifet, A.; Holmes, G.; Pfahringer, B.; Kranen, P.; Kremer, H.; Jansen, T.; and Seidl, T. 2010. Moa: Massive online analysis, a framework for stream classification and clustering. In *Proceedings of the first workshop on applications of pattern analysis*, 44–50. PMLR.

Boracchi, G.; Carrera, D.; Cervellera, C.; and Maccio, D. 2018. QuantTree: Histograms for change detection in multivariate data streams. In *International Conference on Machine Learning*, 639–648. PMLR.

Cobb, O.; and Van Looveren, A. 2022. Context-aware drift detection. In *International conference on machine learning*, 4087–4111. PMLR.

Coelho, R. A.; Torres, L. C. B.; and de Castro, C. L. 2023. Concept Drift Detection with Quadtree-based Spatial Mapping of Streaming Data. *Information Sciences*.

Dau, H. A.; Bagnall, A.; Kamgar, K.; Yeh, C.-C. M.; Zhu, Y.; Gharghabi, S.; Ratanamahatana, C. A.; and Keogh, E. 2019. The UCR time series archive. *IEEE/CAA Journal of Automatica Sinica*, 6(6): 1293–1305.

EP, B. G. 1978. Statistics for Experimenters: An Introduction to Design. *Data Analysis, and Model Building*, 43–48.

Frias-Blanco, I.; del Campo-Ávila, J.; Ramos-Jimenez, G.; Morales-Bueno, R.; Ortiz-Diaz, A.; and Caballero-Mota, Y. 2014. Online and non-parametric drift detection methods based on Hoeffding’s bounds. *IEEE Transactions on Knowledge and Data Engineering*, 27(3): 810–823.

Gama, J.; Medas, P.; Castillo, G.; and Rodrigues, P. 2004. Learning with drift detection. In *Advances in Artificial Intelligence—SBIA 2004: 17th Brazilian Symposium on Artificial Intelligence, Sao Luis, Maranhao, Brazil, September 29-October 1, 2004. Proceedings 17*, 286–295. Springer.

Harries, M. 1999. Splice-2 comparative evaluation: electricity pricing (technical report unsw-cse-tr-9905). *Artificial Intelligence Group, School of Computer Science and Engineering, The University of New South Wales, Sydney*, 2052.

Huggard, H.; Koh, Y. S.; Dobbie, G.; and Zhang, E. 2020. Detecting concept drift in medical triage. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 1733–1736.

Hulten, G.; Spencer, L.; and Domingos, P. 2001. Mining time-changing data streams. In *Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining*, 97–106.

Ikonomovska, E. 2011. Airline dataset. *URL http://kt. ijs. si/elena_ikonomovska/data. html.(Accessed on 02/06/2020)*.

Ismail Fawaz, H.; Forestier, G.; Weber, J.; Idoumghar, L.; and Muller, P.-A. 2019. Deep learning for time series classification: a review. *Data Mining and Knowledge Discovery*, 33(4): 917–963.

Jiang, M.; Wang, Z.; and Dou, Q. 2022. Harmofl: Harmonizing local and global drifts in federated learning on heterogeneous medical images. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 1087–1095.

Jiao, B.; Guo, Y.; Yang, C.; Pu, J.; Zheng, Z.; and Gong, D. 2022. Incremental Weighted Ensemble for Data Streams with Concept Drift. *IEEE Transactions on Artificial Intelligence*.

Li, W.; Yang, X.; Liu, W.; Xia, Y.; and Bian, J. 2022. Dddda: Data distribution generation for predictable concept drift adaptation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 4092–4100.

Liu, A.; Lu, J.; and Zhang, G. 2020. Concept drift detection via equal intensity k-means space partitioning. *IEEE transactions on cybernetics*, 51(6): 3198–3211.

Liu, A.; Song, Y.; Zhang, G.; and Lu, J. 2017. Regional concept drift detection and density synchronized drift adaptation. In *IJCAI International Joint Conference on Artificial Intelligence*.

Lu, J.; Liu, A.; Dong, F.; Gu, F.; Gama, J.; and Zhang, G. 2018a. Learning under concept drift: A review. *IEEE transactions on knowledge and data engineering*, 31(12): 2346–2363.

Lu, J.; Liu, A.; Song, Y.; and Zhang, G. 2020. Data-driven decision support under concept drift in streamed big data. *Complex & intelligent systems*, 6(1): 157–163.

Lu, J.; Xuan, J.; Zhang, G.; and Luo, X. 2018b. Structural property-aware multilayer network embedding for latent factor analysis. *Pattern Recognition*, 76: 228–241.

Miyaguchi, K.; and Kajino, H. 2019. Cogra: Concept-drift-aware stochastic gradient descent for time-series forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, 4594–4601.

Mourtada, J.; Gaiffas, S.; and Scornet, E. 2021. AMF: Aggregated Mondrian forests for online learning. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 83(3): 505–533.

Park, J.; Shalit, U.; Schölkopf, B.; and Muandet, K. 2021. Conditional distributional treatment effect with kernel conditional mean embeddings and U-statistic regression. In *International Conference on Machine Learning*, 8401–8412. PMLR.

Raab, C.; Heusinger, M.; and Schleif, F.-M. 2020. Reactive soft prototype computing for concept drift streams. *Neurocomputing*, 416: 340–351.

Sebastião, R.; and Fernandes, J. M. 2017. Supporting the page-hinkley test with empirical mode decomposition for change detection. In *International Symposium on Methodologies for Intelligent Systems*, 492–498. Springer.

Silverman, B. W. 2018. *Density Estimation for Statistics and Data Analysis*. Monographs on statistics and applied probability. Boca Raton, FL: CRC Press. ISBN 1-351-45617-2.

&lt;page_number&gt;19131&lt;/page_number&gt;

---


## Page 9

Song, X.; Wu, M.; Jermaine, C.; and Ranka, S. 2007. Statistical change detection for multi-dimensional data. In *Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining*, 667–676.

Souza, V. M.; Parmezan, A. R.; Chowdhury, F. A.; and Mueen, A. 2021. Efficient unsupervised drift detector for fast and high-dimensional data streams. *Knowledge and Information Systems*, 63: 1497–1527.

Stucchi, D.; Rizzo, P.; Folloni, N.; and Boracchi, G. 2023. Kernel quanttree. In *International Conference on Machine Learning*, 32677–32697. PMLR.

Tahmasbi, A.; Jothimurugesan, E.; Tirthapura, S.; and Gibbons, P. B. 2021. Driftsurf: Stable-state/reactive-state learning under concept drift. In *International Conference on Machine Learning*, 10054–10064. PMLR.

Virtanen, P.; Gommers, R.; Oliphant, T. E.; Haberland, M.; Reddy, T.; Cournapeau, D.; Burovski, E.; Peterson, P.; Weckesser, W.; Bright, J.; van der Walt, S. J.; Brett, M.; Wilson, J.; Millman, K. J.; Mayorov, N.; Nelson, A. R. J.; Jones, E.; Kern, R.; Larson, E.; Carey, C. J.; Polat, İ.; Feng, Y.; Moore, E. W.; VanderPlas, J.; Laxalde, D.; Perktold, J.; Cimrman, R.; Henriksen, I.; Quintero, E. A.; Harris, C. R.; Archibald, A. M.; Ribeiro, A. H.; Pedregosa, F.; van Mulbregt, P.; and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17: 261–272.

Wan, K.; Liang, Y.; and Yoon, S. 2024. Online Drift Detection with Maximum Concept Discrepancy. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

Wang, K.; Lu, J.; Liu, A.; Song, Y.; Xiong, L.; and Zhang, G. 2022. Elastic gradient boosting decision tree with adaptive iterations for concept drift adaptation. *Neurocomputing*, 491: 288–304.

Wang, K.; Lu, J.; Liu, A.; Zhang, G.; and Xiong, L. 2021. Evolving gradient boost: A pruning scheme based on loss improvement ratio for learning under concept drift. *IEEE Transactions on Cybernetics*.

Yonekawa, K.; Saito, K.; and Kurokawa, M. 2022. RIDEN: Neural-based Uniform Density Histogram for Distribution Shift Detection. In *Proceedings of the Second International Conference on AI-ML Systems*, 1–9.

Yu, E.; Lu, J.; Zhang, B.; and Zhang, G. 2024. Online Boosting Adaptive Learning under Concept Drift for Multistream Classification. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, 16522–16530.

Yuan, L.; Li, H.; Xia, B.; Gao, C.; Liu, M.; Yuan, W.; and You, X. 2022. Recent Advances in Concept Drift Adaptation Methods for Deep Learning. In *IJCAI*, 5654–5661.

Zhou, M.; Lu, J.; Song, Y.; and Zhang, G. 2023. Multi-Stream Concept Drift Self-Adaptation Using Graph Neural Network. *IEEE Transactions on Knowledge and Data Engineering*, 35(12): 12828–12841.

&lt;page_number&gt;19132&lt;/page_number&gt;