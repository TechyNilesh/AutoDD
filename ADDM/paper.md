## Page 1

# AUTOREGRESSIVE BASED DRIFT DETECTION METHOD

Mansour Zoubeirou A Mayaki
Université Côte d’Azur
CNRS, Inria, I3S
Nice France

Michel Riveill
Université Côte d’Azur
CNRS, Inria, I3S
Nice France

&lt;watermark&gt;arXiv:2203.04769v2 [stat.ML] 11 Jun 2023&lt;/watermark&gt;

## ABSTRACT

In the classic machine learning framework, models are trained on historical data and used to predict future values. It is assumed that the data distribution does not change over time (stationarity). However, in real-world scenarios, the data generation process changes over time and the model has to adapt to the new incoming data. This phenomenon is known as concept drift and leads to a decrease in the predictive model’s performance. In this study, we propose a new concept drift detection method based on autoregressive models called **ADDM**. This method can be integrated into any machine learning algorithm from deep neural networks to simple linear regression model. Our results show that this new concept drift detection method outperforms the state-of-the-art drift detection methods, both on synthetic data sets and real-world data sets. Our approach is theoretically guaranteed as well as empirical and effective for the detection of various concept drifts. In addition to the drift detector, we proposed a new method of concept drift adaptation based on the severity of the drift.

**Keywords** Concept drift detection · Data streams · Auto-regressive model · Machine learning · Deep neural networks

## 1 Introduction

Thanks to progress in the field of big data and data analysis, machine learning models and more particularly those based on deep neural networks (Deep Learning) are nowadays experiencing phenomenal success. Since the beginning of the 2010s, neural networks have been developing at high speed and the fields of application are multiplying in all business sectors. In the machine learning framework, models are trained on historical data and used to predict future values. In this framework, we assume that future incoming data streams are stationary, i.e., the data generating process does not change over time. However, this assumption does not hold in most real-world applications [1]. For example the statistical properties of a streaming data can change over time due to seasonality or random events. This phenomenon is known in the machine learning community as concept drift. In the presence of concept drift, the model’s predictions become less accurate over time.

Machine learning models should therefore take in account concept drift and update their weights at the right time. Detecting a concept drift is one of the main challenges when learning with streaming data because of the high speed and their large size sets which are not able to fit in the main memory [2]. To deal with concept drift, many algorithms and methods (ADWIN, DDM, KSWIN, PageHinkley) have been proposed in the literature. Most of these algorithms detect concept drifts by tracking the changes in the model’s error rate or using a distance function to measure the dissimilarity of the input data distribution between some timestamps. These methods are very sensitive to changes leading to large numbers of detected drifts and false alarms [1]. Moreover, most of these algorithms require full and immediate access to ground-truth labels which is an unrealistic assumption in most real-world applications.

To accurately detect concept drifts in stream data, we propose to integrate an autogressive time series model inside the machine learning loop by considering the model’s error as a time series. We used a self-exciting threshold auto-regressive (SETAR) model [3] as the base autoregressive model. SETAR is a nonlinear time series model and a special case of regime switching models in which different models apply to different intervals of values of some key features. Our method has two components: a machine learning model for the learning task and a SETAR model that detects the

---


## Page 2

changes in the learning model’s error rate distribution. We call the new concept drift detection method **ADDM**. This approach can be used with any type of predictive model (Logistic regression, random forest etc.). Our results show that the new concept drift detection method outperforms all state-of-the-art methods on six (6) synthetic data sets and five (5) real-world data sets. ADDM is more accurate and has a very low false alarms rate. A low false alarms rate is very important in real-world application because retraining a machine model is time-consuming and resource-intensive. This method also has some theoretically guarantees as the parameters of the change detection component (SETAR model) are estimated using ordinary least squared (OLS) [4]. Another advantage of our method is that we can construct confidence intervals for the detected drift points using statistical inference and subsampling [5]. In addition to the drift detector, we proposed a new method of concept drift adaptation based on the severity of the drift. The main idea is to aggregate the old and the new models using an estimate of the dissimilarity between the old concept and the new one as weights. The higher is the severity, the less relevant is the old model.

The rest of the paper is outlined as follows. The **Related Works** section discusses the notions related to concept drift and the other studies (or articles) related to concept drifts detection methods/algorithms. The third section is dedicated to the theoretical definition of the self-exciting threshold auto-regressive model and the description of ADDM concept drift detection method. In section **Experimental Data sets**, we describe the data sets used for our experiments. The **hyperparameters optimization** section describes in detail the models architecture, the performance metrics and the drift detection algorithms hyper-parameters optimization. The results are presented and discussed in section **Results and discussions**.

## 2 Related Works

In statistical (or machine) learning domain, concept drift occurs when the statistical properties of the targeted variable $y$ varies arbitrary over time due to a change in the input data X distribution. Concept drifts can be categorised in three groups according to their sources [6]. The first type of concept drift called virtual drift occurs when the data distribution changes but does not affect the decision boundaries: $P_t(X) \neq P_{t+1}(X)$ while $P_t(y|X) = P_{t+1}(y|X)$. Virtual concept drift is not well studied in machine learning community because it does not affect the model’s outputs. The second source of concept drift called actual drift happens when the drift changes the target variable. Thus the a posteriori probability of the data changes in time while it distribution remains unchanged: $P_t(y|X) \neq P_{t+1}(y|X), P_t(X) = P_{t+1}(X)$. The last type of concept drift result from the mixture of the two first sources: $P_t(y|X) \neq P_{t+1}(y|X)$ and $P_t(X) \neq P_{t+1}(X)$. In practice it is very difficult to separate these sources of concept drifts when learning with stream data. The drift detection algorithms just try to detect the changes that affect the model’s output without focusing on their sources.

### 2.1 Concept Drift Understanding

Concept drift understanding answers three main questions: **When** did the drift occur, **How** severe is the change and **Where** are the drifts regions [6]. The **when** refers to the fact that any concept drift detection algorithms should be able to detect the timestamps where the data distribution changes significantly. Recalling the definition of concept drift, when a drift occurs at time $t$, an alarm signal is triggered and it also indicates that the learning system should adapt to a new concept. Another important question is **how** much did the data distribution change at the drift points (severity of the drifts). The severity of concept drift quantifies the dissimilarity between the new concept and the previous concept . The severity is defined as $\Delta = \delta(P_t(X,y), P_{t+1}(X,y))$ where $\delta$ is a function that measures the discrepancy of two data distributions when there is a drift at timestamp $t$ [6]. The greater the value of $\Delta$, the larger the severity of the concept drift. The severity gives an idea of how the learners should adapt to the new concept. If the $\Delta$ is low we may just need to update the learners without changing many parameters. In contrast if the drift is severe, we may need to retrain a whole new model. The last question is to identify **where** the drift regions (new concepts) are located. The drift regions of concept drift are the sub-regions where the new concept and the previous concept are located. These sub-regions are identified by finding parts of the features space where $P_t(X,y)$ and $P_{t+1}(X,y)$ are statistically different. In ensemble learning scenarios, detecting concept drift regions can help predicting instances in stable regions. Moreover, when learning with an artificial neural network we can use the knowledge of the concept drift regions to put weights on the features.

Like all the state-of-the-art concept drift detection algorithms, ADDM responds to the first question. It detects with high precision the drift points by monitoring the model’s error rate. Contrary to the state-of-the-art algorithms, ADDM can compute confidence intervals of each detected drift using statistical hypothesis testing and subsampling methods.

&lt;page_number&gt;2&lt;/page_number&gt;

---


## Page 3

# 2.2 Concept Drift Detection Methods in the Literature

Many algorithms and methods have been proposed in the literature to detect concept drifts in stream data. These methods can be classified into three categories in terms of the test statistics they apply [6]. The first category called **error rate-based drift detection algorithms** refers to all the methods that track changes in the online error rate of base models. The algorithms trigger an alarm when there is a statistically significant increase or decrease of the error rate at some timestamps. Our **ADDM** method belongs to this category. The second category is **data distribution-based drift detection** algorithms which use a distance function or a metric to quantify the dissimilarity between the distribution of data before and after the suspected drift timestamp [6]. These algorithm detect drifts directly from the input data and try to detect the time and the location of the drifts. The last category called **multiple hypothesis test drift detection** methods use multiple hypothesis tests to detect concept drift. The most popular state-of-the-art drift detection algorithms (ADWIN, DDM, KSWIN, PageHinkley) are error rate-based. The **ADaptive WINdowing (ADWIN)** [7] is an adaptive sliding window algorithm for concept drift detection in stream data. **ADWIN** require the user to specify a sensitivity hyperparameter $\alpha \in (0, 1)$ which allows the algorithm to adjust to the input data. A drift is detected when two sub-windows of a recent window of observations exhibit an absolute difference in means larger than $\alpha$. The **Drift Detection Method (DDM)** is a concept drift detection method based on the PAC learning model premise [8]. If the algorithm detects an increase in the error rate higher than a calculated threshold, an alarm is triggered, either change is detected or the algorithm will warn the user that change may occur in the near future. The **Page-Hinkley (PH)** concept drift detector detects changes by computing the observed values and their mean up to the current moment [9]. The algorithm detects a concept drift if the observed mean at some instant is greater than a threshold value $\lambda$. The **Kolmogorov-Smirnov Windowing (KSWIN)** concept drift detection method is based on the Kolmogorov-Smirnov (KS) statistical test [10]. Other versions of these algorithms have been proposed by other authors: Learning with Local Drift Detection (LLDD) [11], Early Drift Detection Method (EDDM) [12], Hoeffding’s inequality based Drift Detection Method (HDDM) [13], Dynamic Extreme Learning Machine (DELM) [14].

Baier et al. [11] used neural network uncertainty instead of the model’s error rate to detect concept drift. The authors proposed to use the **Monte Carlo Dropout** technique to capture the model’s uncertainty. A drift is detected when the model’s uncertainty increases or decreases significantly at some timestamp(s). Their algorithm called Uncertainty Drift Detection (UDD) is based on ADWIN algorithm. The main advantage of UDD method is that in contrast of the majority of error based drift detection algorithms, it does not require full and immediate access to ground-truth labels which is an unrealistic assumption in most real-world use cases [1].

Yan et al. [15] proposed an algorithm based on Hoeffding’s inequality to monitor the error rate and detect concept drift. The main idea of their method is to use Hoeffding’s concentration inequality to examine the consistency of the predictive error. This algorithm relies on the theorem that if the data distribution is stationary if the difference between the predictive error at time $t$ denoted $p_t$ as and its lower bound $p_{bayes}$ goes to 0 when the number of training instances increases [16]. The authors used Hoeffding’s Inequality to estimate the desired upper bound above which the error rate is considered highly unstable. If the predictive error difference $\Delta_t = p_t - p_{bayes}$ is very high after learning a large enough number of instances, it means that their is a concept drift and the data distribution has changed. According to there results, their method gives better results than the state-of-the-art methods. Wang et al. [17] proposed a new concept drift detection method for class imbalanced problem called DDM-OCI. The new method is inspired from the DDM [8], instead of monitoring the model’s error rate, the authors used the recall of the minority class to detect changes in the data distribution. Their results show that, DDM-OCI responds to new concepts faster than the model applying DDM. Greco and Tania [18] proposed a real-time unsupervised per-label drift detection methodology based on embedding distribution distances in deep learning models . Their method exploits the inner representations assigned by a deep learning model to new unseen data to detect drifts.

# 3 Our approach for learning Under Concept Drift

## 3.1 Self-exciting Threshold Autoregressive (SETAR) Models

The self-exciting threshold autoregressive model is a nonlinear time series model proposed by Tong in 1978 [4]. This model is special case of regime switching models in which different models apply to different intervals of values of some key variable. This model has certain properties such as limit cycles, amplitude dependent frequencies, and jump phenomena that can not be captured by classic linear time series models [3]. Formally, a SETAR model with k regimes can be written mathematically as follows [4]:

$$Y_t = \phi_0^{(i)} + \phi_1^{(i)} Y_{t-1} + \cdots + \phi_{p_i}^{(i)} Y_{t-p} + \epsilon_t^i \quad (1)$$

$$r_{i-1} \leq Y_{t-d} < r_i \quad (2)$$

&lt;page_number&gt;3&lt;/page_number&gt;

---


## Page 4

Where $i = 1 \ldots k$, $1 < d \leq \max(p_i)$ a positive integer, $Y_t$ is a time series and $Y_{t-d}$ the threshold variable. The thresholds values are $-\infty < r_0 < r_1 < \cdots < r_k < +\infty$; for each regime $i$, the error term $\epsilon^i_t$ is a sequence of martingale differences satisfying:

$$E(\epsilon^i_t|F_{t-1}) = 0, \quad sup_t = E(|\epsilon^i_t|^{\delta}|F_{t-1}|) < \infty$$

$$a.s. \quad for \quad some \quad \delta > 2$$

With $F_{t-1}$ a $\sigma$ field generated by $\{\epsilon^i_{t-j}|j = 1,\ldots;i = 1,\ldots,k\}$. Such a process partitions the one-dimensional Euclidean space into k regimes and follows a linear auto-regressive model in each regime [4]. A two regime SETAR model can be written as follows:

$$Y_t = (\phi_0 + \phi_1 Y_{t-1} + \cdots + \phi_p Y_{t-p}) \mathbb{1}_{Y_{t-d} \leq r} + (\beta_0 + \beta_1 Y_{t-1} + \cdots + \beta_p Y_{t-p}) \mathbb{1}_{Y_{t-d} > r} + \epsilon_t \quad (3)$$

Where $p$ denotes the autoregressive level , $Y_{t-d}$ the threshold variable and $r$ the threshold parameter. In principle, we would like $\epsilon_t$ to be conditionally heteroskedastic, but for formal theory, we assume that $\epsilon_t$ is $iid (0, \sigma^2)$. This model is called self-exciting because the threshold variable is a function of the past values of the endogenous variable $Y_t$. Since the SETAR model is a locally linear model, ordinary least squares (OLS) techniques can be used to estimate its parameters [19]. Under the assumption that the error $\epsilon_t$ is $iid N(0, \sigma^2)$, OLS is equivalent to maximum likelihood estimation. The model in equation 1 can be rewritten as follows:

$$Y_t = \Phi' X_t(r) + \epsilon_t \quad (4)$$

$$X_t = (1, Y_{t-1}, \ldots, Y_{t-p})'$$

$$X_t(r) = (X_t \mathbb{1}_{Y_{t-d} \leq r}, X_t \mathbb{1}_{Y_{t-d} > r})$$

Where $\Phi = (\phi, \beta)$, $\phi = (\phi_0, \phi_1, \ldots, \phi_p)$ and $(\beta = \beta_0, \beta_1, \ldots, \beta_p)$.

The parameters of interest are $\Phi$ and $r$. For a given threshold value $r$, the OLS estimate of $\Phi$ is [19]

$$\hat{\Phi}(r) = \left( \sum_{t=1}^{T} X_t(r)' X_t(r) \right)^{-1} \left( \sum_{t=1}^{T} X_t(r)' Y_t \right) \quad (5)$$

With estimated residuals $\hat{\epsilon}_t(r) = Y_t - X_t(r)' \hat{\Phi}(r)$ and their estimated variance

$$\hat{\sigma}_T^2(r) = \frac{1}{T} \sum_{t=1}^{T} \hat{\epsilon}_t^2 \quad (6)$$

The estimation task is now reduced to finding the threshold values $r$ that minimizes estimated residuals variance $\hat{\sigma}_T^2(r)$ which depends exclusively on $r$. The other parameters are then computed by using equation (5). The threshold parameter is computed by minimizing equation (6) as follows:

$$\hat{r} = \arg \min_{r \in R} \hat{\sigma}_T^2(r) \quad (7)$$

Where $R = \{y_{t-d}, \text{for all } d+1 \leq t \leq T\}$. $R$ has a finite number of elements $[T-(d+1)]$.

Since the model’s parameters are estimated using ordinary least squared (OLS) method, we have some theoretical guarantees of its convergence [4]. We can also construct confidence intervals for the detected drift points by using statistical inference and sub-sampling [5].

### 3.2 ADDM for Concept Drift Detection

Our **ADDM** method belongs to the category of concept drift detection methods based on error rate monitoring. The model triggers an alarm when there is a statistically significant increase or decrease of the error rate at some timestamps. As proved by Gama et al. [16], if the data distribution is stationary, the error rate should converge to its minimum value. Therefore if the error rate becomes very high after predicting a large enough sample, it may indicate that the data distribution has changed and the model is no longer fit to the data. ADDM has two components: a machine learning

&lt;page_number&gt;4 4&lt;/page_number&gt;

---


## Page 5

model for the prediction task and a SETAR model that detects the changes in the learning model’s error rate $Y_t$. When new data instances arrive, they are predicted at the time of arrival with the deep learning model. The prediction errors are then computed and used as target variable $Y_t$ for the SETAR model (see Fig.1). Referring to the definition of the SETAR model in equation (1), our model forecasts the error rate using a linear combination of its past values assuming that the behavior of the error rate changes once it enters a different regime or concept. The threshold values estimated by equation (7) correspond to the concept drift points in the data. We can also construct confidence intervals for each detected drift using statistical inference and sub-sampling [5].

&lt;img&gt;
A diagram showing the architecture and dataflow of ADDM.
- A cloud labeled "Old data" feeds into a box labeled "Train model".
- From the "Train model" box, an arrow goes to a box labeled "Model".
- An arrow from the "Model" box goes to a box labeled "Drift detector".
- An arrow from the "Drift detector" box goes to a box labeled "Output".
- An arrow from the "Drift detector" box goes back to the "Model" box, labeled "Update model".
- An arrow from the "Model" box goes to a dashed box labeled "Train new model".
- An arrow from the "Train new model" box goes back to the "Model" box, labeled "Drift detected".
- An arrow from the "Drift detector" box goes to the "Output" box.
- An arrow from the "Drift detector" box goes to the "Model" box, labeled "Incoming data for inference".
- An arrow from the "Drift detector" box goes to a cloud labeled "New data".
&lt;/img&gt;

Figure 1: Architecture and Dataflow of ADDM. ① Train a predictive model with old data set. ② Predict new incoming data and compute the error rate. ③ Search for changes in the model’s error rate. If drift is detected, triggers a signal to update the model. ④ Train new model with most recent data if drift is detected and update the old model. ⑤ Predict the final output with updated model.

Figure 1: Architecture and Dataflow of ADDM. (1) Train a predictive model with old data set. (2) Predict new incoming data and compute the error rate. (3) Search for changes in the model’s error rate. If drift is detected, triggers a signal to update the model. (4) Train new model with most recent data if drift is detected and update the old model. (5) Predict the final output with updated model.

## 3.3 Model Updating or Concept drift adaptation

After detecting concept drifts, we need to update the model so that it adapts to the new data distribution. In the literature there are mainly three groups of drift adaptation methods. The first method consist of retraining a whole new model when a concept drift is detected. The second strategy is ensemble method witch consist of aggregating a new model trained on the samples from the new distribution with the old models. This strategy can save significant effort to retrain a new model for recurring concepts. Ensemble methods comprise a set of base classifiers that may have different types or different parameters [6]. The last drift adaptation strategy consists of partially updating the model when the underlying data distribution changes. This strategy is more efficient than retraining an entire model when the drift only occurs in well located regions (decision tree algorithm are suited in this case because trees have the ability to examine and adapt to each sub-region separately). This approach can be difficult to use in case of deep learning models because these models are considered as black boxes and we don’t know witch parameters to update.

We propose a new concept drift adaptation method that consist of aggregating the old model with a new model trained on the most recent samples using the severity of the concept drift. The main idea is aggregate the old and new model using an estimate of the dissimilarity (severity of the drift) between the old concept and the new one as weight. The higher is the severity, the less relevant is the old model. The severity denoted $w_t$ is also used as the weight of the new model in the final model. We first compute the third quantile $Q_3$ in each regime and compute $w_t$ as follows:

$$w_t = \frac{\max(Q^0_3, Q^t_3)}{Q^0_3 + Q^t_3}$$ (8)

&lt;page_number&gt;5&lt;/page_number&gt;

---


## Page 6

Where $Q_3^0$ is the third quantile of the error rate in the old concept and $Q_3^t$ in the new concept. We used the third quantile $Q_3$ just to make sure that we have a good estimate of the error rate in each regime. One can use any other quantile or aggregating metrics (mean, variance etc.). The quantiles are more suited because they are less sensitive to extreme values. The term $\max(Q_3^0, Q_3^t)$ is used to ensure that the model learned under the new concept always gets the highest weight during aggregation. The final model is defined as follows:

$$M_{\hat{\phi}}^t \leftarrow M_{\hat{\phi}}^0 \cdot (1 - w_t) + w_t \cdot M_{new}$$  (9)

Where $M_{\hat{\phi}}^0$ is the old model and $M_{new}$ the new model trained with the most recent data. The new model $M_{new}$ can be learned on a subset (or a window) containing the most recent data or on the whole data set. The main advantage of our method is that it takes into account the severity of the drift when updating the model. If the drift is very severe, the new model has a much more important role than the old one and the influence of the old model may fade. Another advantage is that, it’s very flexible and can be used with any kind of model. For example when learning with artificial neural networks, we can average the old and the new model parameters using $w_t$ or just average their outputs.

**ADDM** algorithm is defined as follows:

**Algorithm 1 ADDM algorithm**

1: **Input:** Training data $D_{tr}$; Validation data $D_{val}$; Data stream $D_s$
2: Train deep learning model: $M_{\hat{\phi}}^0 \leftarrow M_{\hat{\phi}} \cdot fit(D_{tr}, D_{val})$
3: Compute validation error: $\hat{e}_{val} \leftarrow (y_{val} - \hat{y}_{val})^2$
4: Fixe a time window $w$
5: **repeat**
6: Receive incoming data instances $x_{t-w}$
7: Predict values: $\hat{y}_{t-w} \leftarrow M_{\hat{\phi}}^0 \cdot predict(x_{t-w})$
8: Compute error: $\hat{e}_{t-w} \leftarrow (y_{t-w} - \hat{y}_{t-w})^2$
9: Learn Setar model with $\hat{e}_{t-w} \cup \hat{e}_{val}$
10: **if change is detected then**
11: Compute drift severity: $w_t \leftarrow \frac{\max(Q_3^0, Q_3^t)}{Q_3^0 + Q_3^t}$
12: Get most recent labeled data $D_{recent}$
13: Train new model $M_{new} \leftarrow M_{\hat{\phi}} \cdot fit(D_{recent})$
14: $M_{\hat{\phi}}^t \leftarrow M_{\hat{\phi}}^0 \cdot (1 - w_t) + w_t \cdot M_{new}$
15: **end if**
16: **until** $D_s$ ends

## 4 Experimental Data sets

In order to evaluate our method’s capabilities, we compared its performance to those of seven (7) state-of-the-art methods on six (6) synthetic data sets with artificial concept drifts and five (5) real-world data sets. The synthetic data sets were simulated using the python scikit-multiflow package [20]. The **Friedman** multi-variate regression data set [21] consist of teen features each generated from a uniform distribution from the interval [0, 1]. The Friedman data set is commonly used to test concept drift detection methods . In our experiments, we simulated three different versions of the Friedman data sets with different types of drifts. The **Brieman** regression data set is inspired by Baeir et al. [1]. The data set contains teen features, simulated from uniform distribution. The **Mixed** data set was inspired by Gama et al. [8] and has 6 attributes. Four attributes are relevant for classification: two boolean attributes and two numeric attributes uniformly distributed from 0 to 1 [20]. The **Agrawal** stream generator was first introduced by Agrawal et al. [22]. The generator generates a stream data set of nine features, six numeric and three categorical [20] for binary classification task. We generated two data sets from this generator with different types of concept drift. At the end we have six synthetic data sets among which three regression data sets and three classification data sets. In each data set, artificial concept drifts were introduced by modifying the distribution of some features.

In addition to these synthetic data sets, we tested our method on five real-world data sets. Note that all the following data sets are publicly available on UCI Machine Learning Repository website [23]. The **Panama electricity** data set contains historical records of Panama’s electricity demand and weather measures from January 2015 until June 2020 [24]. The data set contains historical electricity load, calendar information related to holidays (and school period) and Weather variables, such as temperature, relative humidity, precipitation, and wind speed, from three main cities in Panama [24]. The goal is to predict the electricity demand using all available features. In this data set, concept drift is

&lt;page_number&gt;6&lt;/page_number&gt;

---


## Page 7

present due to seasonal weather changes which affects the electricity demand. The **Italian air quality** data set contains the responses of a gas multi-sensor devices deployed on the field in the Italian main cities. Hourly responses averages are recorded along with gas concentrations references from a certified analyzer [25]. The data were recorded from March 2004 to February 2005. They recorded some air quality measures such as the hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2). The goal is to predict the benzene concentration (C6H6(GT)), which is a proxy for air pollution. Like in the electricity data set, concept drift is present due to seasonal weather changes. According to the World Air Quality Index project, the air is very polluted in the Italian main cities between November and February [26]. The **NSW** data set contains data from the Australian New South Wales electricity market [8]. In this market, prices are flexible and are affected by demand and supply of the market. The data set contains 45.312 instances and nine features dated from 7 May 1996 to 5 December 1998. The goal is to predict if the electricity price goes up or down each 30 minutes. Concept drift is present due to seasonal weather changes which affects the electricity demand and its price. The **gas sensor array drift** data set contains measurements from 16 chemical sensors utilized in simulations for drift compensation in a discrimination task of 6 gases at various levels of concentrations. The data set was gathered over a period of 36 months in a gas delivery platform facility situated [27]. The goal is to achieve good classification performance over time. Concept drifts are present in the data due to sensor aging and external alterations. The **Beijing Multi-Site** air quality data set contains 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing. Each variable is measured hourly from March 1st, 2013 to February 28th, 2017. The goal is to predict the PM2.5 variable which is a proxy of air quality measure. Concept drifts are present in this data set due to seasonal weather changes.

## 5 Experimental Design

### 5.1 Performance Metrics

In the case of synthetic data sets we know exactly where the drifts occurred so we can compare the detector’s outputs to them. For the synthetic data sets, we use the following metrics: detection accuracy, True positive, false positive (or false alarms) and the mean time to detection (MTD). Contrary to synthetic data sets, real-world data sets don’t have specified concept drift points. It is therefore very difficult to evaluate concept drift detection algorithms on them. In this case, we can’t use metrics like accuracy or true positive rate to compare the detectors. To compare ADDM to the state-of-the-art algorithms on real-world data sets, we use the mean squared error (MSE) loss of the learning model in case of regression task and the cross entropy loss for classification tasks. For each data set, if a drift is detected, a new model is learned from scratch and evaluated on a subset of the most recent data. The final performance of the detector is computed by averaging its losses on all the detected regions. For each detection algorithm, we also take in account the number of detections because in real-world applications a detector that gives a large number of alarms is not optimal.

### 5.2 Hyper-parameters Optimization

In this article, we have used deep learning based models as the backbone of the prediction step of the ADDM method, but any type of machine learning model can be used (logistic regression, random forest, etc.). For the synthetic data sets, our used simple multi-layer perceptron (MLP) neural networks with two hidden layers. In case of real-world data sets, we used long short-term memory (LSTM) neural networks architecture (see Table 1). Each model is trained and validated on a subset of the data set where there is no concept drift. The learned model is then used to predict values for new incoming data where concept drift is suspected to be present. We then compute the error rate of the model on new data and try to find if there are significant changes at some timestamps. When learning with artificial neural networks, instead of monitoring the error rate which requires total access to the true labels, we can use the model’s uncertainty to detect concept drift as done by Baeir et al. [1]. Using Monte Carlo Dropout, we can compute and monitor the model’s uncertainty and detect concept drifts.

The state-of-the-art drift detection methods and algorithms (DDM, ADWIN, PageHinkley, HDDM,KSWIN) used in this study have some hyper-parameters that should be well chosen carefully so the algorithm can adjust to the data set. For each method/algorithms, we determined the optimal hyper-parameters by using a subset of the data set that we called experimental set. For each synthetic data set, we took a subset containing one concept drift as the experimental set. Each algorithm is executed on the experimental sets to find the best hyper-parameters. These hyper-parameters are then used in the final experiments. We compared these state-of-the-art methods to our ADDM drift detection method. As described in section 3.1, the SETAR model requires the user to set some hyper-parameters. The main parameters are the time delay for the threshold variable $d$, the auto regressive level $p$. In our study the parameters values are: $d = 2$ and $p = 5$. We aim to evaluate and give a fair comparison among the detectors concerning the performances of real concept drift detection. The optimal hyper-parameters are listed in table 1

&lt;page_number&gt;7&lt;/page_number&gt;

---


## Page 8

Table 1: Description and Neural network architecture of each data set. Drifts refers to the number of drifts.The remaining columns show, the optimal hyper-parameter according to the detector.

<table>
<thead>
<tr>
<th>Data set</th>
<th>Samples</th>
<th>Features</th>
<th>Target</th>
<th>Drifts</th>
<th>architecture</th>
<th>Hyper-parameters</th>
<th>Epochs</th>
<th>ADWIN</th>
<th>Page-Hinkley</th>
<th>KSWIN</th>
</tr>
</thead>
<tbody>
<tr>
<td>Friedman</td>
<td>20000</td>
<td>3</td>
<td>continuous</td>
<td>3</td>
<td>MLP</td>
<td>(30,15,1)</td>
<td>50</td>
<td>$1 \times 10^{-4}$</td>
<td>$1 \times 10^{-6}$</td>
<td>0.001</td>
</tr>
<tr>
<td>Friedman no return</td>
<td>20000</td>
<td>3</td>
<td>continuous</td>
<td>6</td>
<td>MLP</td>
<td>(30,15,1)</td>
<td>50</td>
<td>$1 \times 10^{-6}$</td>
<td>$1 \times 10^{-6}$</td>
<td>0.0034</td>
</tr>
<tr>
<td>Brieman 2d planes</td>
<td>20000</td>
<td>11</td>
<td>continuous</td>
<td>6</td>
<td>MLP</td>
<td>(30,15,1)</td>
<td>50</td>
<td>$1 \times 10^{-3}$</td>
<td>$1 \times 10^{-6}$</td>
<td>0.0032</td>
</tr>
<tr>
<td>Agrawal 32</td>
<td>20000</td>
<td>3</td>
<td>2 classes</td>
<td>1</td>
<td>MLP</td>
<td>(20,10,1)</td>
<td>50</td>
<td>$5 \times 10^{-6}$</td>
<td>50</td>
<td>0.0034</td>
</tr>
<tr>
<td>Agrawal 3213</td>
<td>20000</td>
<td>3</td>
<td>2 classes</td>
<td>4</td>
<td>MLP</td>
<td>(20,10,1)</td>
<td>50</td>
<td>$4.2 \times 10^{-5}$</td>
<td>50</td>
<td>0.0029</td>
</tr>
<tr>
<td>Mixed</td>
<td>20000</td>
<td>5</td>
<td>continuous</td>
<td>3</td>
<td>MLP</td>
<td>(20,10,1)</td>
<td>50</td>
<td>0.0441</td>
<td>$1 \times 10^{-6}$</td>
<td>0.0059</td>
</tr>
<tr>
<td>Panama electricity</td>
<td>48048</td>
<td>17</td>
<td>continuous</td>
<td>-</td>
<td>LSTM</td>
<td>(120,60,30)</td>
<td>50</td>
<td>$1 \times 10^{-5}$</td>
<td>$1 \times 10^{-10}$</td>
<td>$1 \times 10^{-10}$</td>
</tr>
<tr>
<td>Italian air quality</td>
<td>8991</td>
<td>14</td>
<td>continuous</td>
<td>-</td>
<td>LSTM</td>
<td>(64,32,15)</td>
<td>50</td>
<td>$3.41 \times 10^{-4}$</td>
<td>$1 \times 10^{-10}$</td>
<td>$1 \times 10^{-10}$</td>
</tr>
<tr>
<td>NSW electricity</td>
<td>45312</td>
<td>9</td>
<td>2 classes</td>
<td>-</td>
<td>LSTM</td>
<td>(120,60,30)</td>
<td>50</td>
<td>$1 \times 10^{-5}$</td>
<td>$1 \times 10^{-10}$</td>
<td>$1 \times 10^{-10}$</td>
</tr>
<tr>
<td>Gas sensor drift</td>
<td>13910</td>
<td>129</td>
<td>6 classes</td>
<td>-</td>
<td>MLP</td>
<td>(120,120,6)</td>
<td>50</td>
<td>$4 \times 10^{-15}$</td>
<td>0.05</td>
<td>$1 \times 10^{-10}$</td>
</tr>
<tr>
<td>Beijing air quality</td>
<td>420768</td>
<td>18</td>
<td>continuous</td>
<td>-</td>
<td>LSTM</td>
<td>(120,60,30)</td>
<td>50</td>
<td>$6 \times 10^{-5}$</td>
<td>$1 \times 10^{-10}$</td>
<td>$1 \times 10^{-10}$</td>
</tr>
</tbody>
</table>

# 6 Results and Discussions

In this section, we present the results of the state-of-the-art drift detection methods and ADDM method on the experimental data sets. For each data set, we compared ADDM to seven state-of-the-art concept drift detection methods.

## 6.1 Results on Synthetic Data sets

Table 2 shows the experimental results of the drift detection algorithms on the synthetic data sets. Recall that in case of synthetic data sets we know the exact drift points so we can compare them to the detector’s outputs. These results show that ADDM outperforms all other methods in terms of true positives (TP) and false alarms (FA). It has a very low false alarm rate. Despite the parameter optimization , note that the state-of-the-art methods detect a very large number of drifts. This illustrates these algorithms’ problem of high reactivity leading to a large number of false positive drift detection [1]. Contrary to state-of-the-art methods, ADDM is less sensitive to small variation and only detects statistically significant drifts. This is very important in real-world application because retraining a machine model is time-consuming and resource-intensive. Fig. 2a and Fig. 2b show the results of all the algorithms applied respectively to the Mixed and the Brieman 2d planes data sets. On these data sets, ADDM accurately detects all the drifts unlike the other algorithms. The Mixed data set contains three concept drift points and ADDM is the only method capable of accurately detecting all the drift points (horizontal line with red plus (+) markers in Fig. 2a). The Brieman data set contains six (6) drift points, ADDM accurately detected five of them (horizontal line with red plus (+) markers in Fig. 2b) when none of the other methods detects more than one drift.

Table 2: Detector performance on synthetic data sets. The columns show the false alarms (FA), true positives (TP) and Mean Time to Detection (MTD) of the detector. The MTD is expressed in seconds

<table>
<thead>
<tr>
<th>Method</th>
<th>Metric</th>
<th>Agrawal 32</th>
<th>Agrawal 3213</th>
<th>Mixed</th>
<th>Friedman</th>
<th>fridman no return</th>
<th>Brieman 2d planes</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">ADDM</td>
<td>TP</td>
<td>1</td>
<td>2</td>
<td>3</td>
<td>3</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>FA</td>
<td>0</td>
<td>2</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>MTD</td>
<td>1.5</td>
<td>3.657</td>
<td>0.368</td>
<td>21.823</td>
<td>11.912</td>
<td>20.711</td>
</tr>
<tr>
<td rowspan="3">ADWIN</td>
<td>TP</td>
<td>0</td>
<td>4</td>
<td>2</td>
<td>1</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>FA</td>
<td>7</td>
<td>15</td>
<td>11</td>
<td>7</td>
<td>10</td>
<td>2</td>
</tr>
<tr>
<td>MTD</td>
<td>0.342</td>
<td>0.657</td>
<td>0.128</td>
<td>1.182</td>
<td>0.935</td>
<td>1.289</td>
</tr>
<tr>
<td rowspan="3">DDM</td>
<td>TP</td>
<td>0</td>
<td>2</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>FA</td>
<td>17</td>
<td>21</td>
<td>15</td>
<td>6</td>
<td>16</td>
<td>5</td>
</tr>
<tr>
<td>MTD</td>
<td>0.372</td>
<td>0.612</td>
<td>0.126</td>
<td>1.435</td>
<td>0.916</td>
<td>1.348</td>
</tr>
<tr>
<td rowspan="3">KSWIN</td>
<td>TP</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>FA</td>
<td>18</td>
<td>21</td>
<td>19</td>
<td>4</td>
<td>9</td>
<td>6</td>
</tr>
<tr>
<td>MTD</td>
<td>0.398</td>
<td>0.690</td>
<td>0.178</td>
<td>1.383</td>
<td>1.103</td>
<td>1.287</td>
</tr>
<tr>
<td rowspan="3">PH</td>
<td>TP</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>FA</td>
<td>15</td>
<td>19</td>
<td>17</td>
<td>3</td>
<td>11</td>
<td>7</td>
</tr>
<tr>
<td>MTD</td>
<td>0.411</td>
<td>0.613</td>
<td>0.138</td>
<td>1.229</td>
<td>0.980</td>
<td>1.360</td>
</tr>
<tr>
<td rowspan="3">EDDM</td>
<td>TP</td>
<td>0</td>
<td>5</td>
<td>3</td>
<td>0</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>FA</td>
<td>14</td>
<td>22</td>
<td>17</td>
<td>4</td>
<td>17</td>
<td>3</td>
</tr>
<tr>
<td>MTD</td>
<td>0.355</td>
<td>0.611</td>
<td>0.134</td>
<td>1.336</td>
<td>1.044</td>
<td>1.247</td>
</tr>
<tr>
<td rowspan="3">HDDM_A</td>
<td>TP</td>
<td>0</td>
<td>2</td>
<td>3</td>
<td>0</td>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>FA</td>
<td>17</td>
<td>23</td>
<td>15</td>
<td>5</td>
<td>17</td>
<td>3</td>
</tr>
<tr>
<td>MTD</td>
<td>0.384</td>
<td>0.819</td>
<td>0.140</td>
<td>1.373</td>
<td>1.012</td>
<td>1.388</td>
</tr>
<tr>
<td rowspan="3">HDDM_W</td>
<td>TP</td>
<td>0</td>
<td>1</td>
<td>3</td>
<td>0</td>
<td>0</td>
<td>3</td>
</tr>
<tr>
<td>FA</td>
<td>17</td>
<td>19</td>
<td>18</td>
<td>6</td>
<td>16</td>
<td>3</td>
</tr>
<tr>
<td>MTD</td>
<td>0.403</td>
<td>0.630</td>
<td>0.133</td>
<td>1.343</td>
<td>1.236</td>
<td>1.324</td>
</tr>
</tbody>
</table>

&lt;page_number&gt;8&lt;/page_number&gt;

---


## Page 9

&lt;img&gt;
A line chart showing error rate over time (in seconds) for different drift detection methods on a Mixed data set.
The y-axis ranges from 0.0 to 2.5, representing Error rate or uncertainty.
The x-axis represents Time (s), ranging from 0 to approximately 2000.

Legend:
- error rate
- ADDM (red plus (+) markers)
- ADWIN (blue asterisk markers)
- DDM (orange diamond markers)
- PH (yellow star markers)
- KSWIN (red circle markers)
- actual drifts (green dashed line with dots)

Key points on the graph:
- The error rate starts near 0.0 at t=0.
- Around t=500, the error rate increases sharply, reaching about 1.0.
- Between t=500 and t=1000, the error rate fluctuates around 1.0.
- Around t=1500, the error rate drops below 1.0 and continues to fluctuate.
- Around t=2000, the error rate stabilizes near 1.8.

The green dashed line with dots indicates the actual drifts, showing two main clusters of drifts:
- One cluster around t=500-1000, with about 13 detections.
- Another cluster around t=1500-2000, with about 6 detections.

The horizontal lines show the detected drifts by each detector:
- KSWIN detects 19 drifts.
- PH detects 18 drifts.
- DDM detects 16 drifts.
- ADWIN detects 13 drifts.
- ADDM detects 3 drifts.
&lt;/img&gt;

(a) Detected drifts on the **Mixed** data set. The vertical lines in green are the actual drifts points. The horizontal lines show the detected drifts by each detector. The horizontal line with red plus (+) markers shows our method’s detections.

---

&lt;img&gt;
A line chart showing error rate over time (in seconds) for different drift detection methods on a Brieman 2d planes data set.
The y-axis ranges from 0.0 to 2.5, representing Error rate or uncertainty.
The x-axis represents Time (s), ranging from 0 to approximately 17500.

Legend:
- error rate
- ADDM (red plus (+) markers)
- ADWIN (blue asterisk markers)
- DDM (orange diamond markers)
- PH (yellow star markers)
- KSWIN (red circle markers)
- actual drifts (green dashed line with dots)

Key points on the graph:
- The error rate starts near 0.0 at t=0.
- Around t=2500, the error rate increases sharply, reaching about 1.5.
- Between t=2500 and t=5000, the error rate fluctuates around 1.5.
- Around t=7500, the error rate drops below 1.0 and continues to fluctuate.
- Around t=15000, the error rate stabilizes near 1.2.

The green dashed line with dots indicates the actual drifts, showing one main cluster of drifts:
- One cluster around t=2500-5000, with about 6 detections.

The horizontal lines show the detected drifts by each detector:
- KSWIN detects 7 drifts.
- PH detects 8 drifts.
- DDM detects 6 drifts.
- ADWIN detects 2 drifts.
- ADDM detects 6 drifts.
&lt;/img&gt;

(b) Detected drifts on the **Brieman 2d planes** data set. The vertical lines in green are the actual drifts points. The horizontal lines show the detected drifts by each detector. The horizontal line with red plus (+) markers shows our method’s detections.

Figure 2: Detector performance on synthetic data sets

## 6.2 Results on Real-world Data sets

Table 3 shows the results of the drift detection algorithms on real-world data sets. Recall that when comparing ADDM to state-of-the-art algorithms on real-world data sets, we use the mean squared error loss of the learning model in case of regression task and the cross entropy loss for classification tasks. In order to have a better view of the detector’s performances, we also listed the number of times the model was retrained (nb_train). The number of retraining is equal to the number of detected drifts and gives us an idea of how sensitive the detector is. As expected, the state-of-the-art detectors lead to a large number of retraining. By combining the loss and the number of retraining, in almost all cases ADDM outperforms the state-of-the-art detectors. In the rare cases where other methods have outperformed ADDM, the improvement is very small and the retraining is at least two times that of ADDM. For example, on the Gas sensor drift data set, the KSWIN algorithm retrained the model 42 times (loss=1.69) while ADDM retrained the model only 7 times (loss=1.89).

Table 3: Performance of detectors on real-world data sets. The model’s loss (the lower the best) and the number of retrainings nb_train (the lower the less computationally expensive).

<table>
<thead>
<tr>
<th>Method</th>
<th>Metric</th>
<th>Panama electricity</th>
<th>Italian air quality</th>
<th>Beijing air quality</th>
<th>Gas sensor drift</th>
<th>NSW data set</th>
</tr>
</thead>
<tbody>
<tr>
<td>ADDM</td>
<td><em>loss</em></td>
<td><strong>0.020</strong></td>
<td><strong>0.007</strong></td>
<td><strong>0.006</strong></td>
<td><strong>1.89</strong></td>
<td><strong>0.290</strong></td>
</tr>
<tr>
<td></td>
<td><em>nb_retrain</em></td>
<td>7</td>
<td>5</td>
<td>5</td>
<td>7</td>
<td>5</td>
</tr>
<tr>
<td>ADWIN</td>
<td>loss</td>
<td>0.032</td>
<td>0.02</td>
<td>0.016</td>
<td>1.727</td>
<td>0.234</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>7</td>
<td>0</td>
<td>195</td>
<td>21</td>
<td>12</td>
</tr>
<tr>
<td>DDM</td>
<td>loss</td>
<td>0.030</td>
<td>0.014</td>
<td>0.014</td>
<td>1.714</td>
<td>0.253</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>18</td>
<td>4</td>
<td>145</td>
<td>41</td>
<td>17</td>
</tr>
<tr>
<td>KSWIN</td>
<td>loss</td>
<td>0.034</td>
<td>0.011</td>
<td>0.016</td>
<td>1.69</td>
<td>0.255</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>10</td>
<td>6</td>
<td>132</td>
<td>42</td>
<td>10</td>
</tr>
<tr>
<td>PH</td>
<td>loss</td>
<td>0.044</td>
<td>0.016</td>
<td>0.016</td>
<td>1.724</td>
<td>0.239</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>12</td>
<td>7</td>
<td>119</td>
<td>37</td>
<td>11</td>
</tr>
<tr>
<td>EDDM</td>
<td>loss</td>
<td>0.026</td>
<td>0.008</td>
<td>0.004</td>
<td>1.748</td>
<td>0.276</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>10</td>
<td>3</td>
<td>78</td>
<td>33</td>
<td>8</td>
</tr>
<tr>
<td>HDDM_A</td>
<td>loss</td>
<td>0.058</td>
<td>0.007</td>
<td>0.003</td>
<td>1.694</td>
<td>0.260</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>5</td>
<td>4</td>
<td>69</td>
<td>23</td>
<td>9</td>
</tr>
<tr>
<td>HDDM_W</td>
<td>loss</td>
<td>0.059</td>
<td>0.009</td>
<td>0.004</td>
<td>1.696</td>
<td>0.288</td>
</tr>
<tr>
<td></td>
<td>nb_retrain</td>
<td>5</td>
<td>3</td>
<td>58</td>
<td>27</td>
<td>8</td>
</tr>
</tbody>
</table>

&lt;page_number&gt;9&lt;/page_number&gt;

---


## Page 10

# Conclusion

Detecting concept drift is important in real-world applications as it leads to a decrease in machine learning models performance. The traditional concept drift detection methods are very sensitive to changes and leads to a large number of false alarms. These methods also often require full access to the true labels. In this paper, we propose a method that combines a machine learning algorithm with autoregressive time series models to detect concept drift in stream data. The main idea is to consider the error rate of a machine learning model as a time series and model them with an autoregressive time series model. We compared ADDM to seven (7) state-of-the-art concept drift detection algorithms on six (6) synthetic data sets and five (5) real-world data sets. The results show that it outperforms all the state-of-the-art algorithms in terms of accuracy and has a very low false alarm rate. In addition to the drift detection method, we proposed a new method of concept drift adaptation based on the severity of the drift. The main idea is to aggregate the old and new model using an estimate of the dissimilarity between the old concept and the new one as weights. The higher is the severity, the less relevant is the old model. In future works we aim to use auto-regressive models to detect concept drifts using directly the input data instead of the error rate or the model’s uncertainty. This can be very useful in cases where the true labels are not available.

# References

[1] Lucas Baier, Tim Schlör, Jakob Schöffer, and Niklas Kühl. Detecting concept drift with neural network model uncertainty. *arXiv preprint arXiv:2107.01873*, 2021.
[2] Shweta Kadam. A survey on classification of concept drift with stream data. 2019.
[3] Howell Tong and Keng S Lim. Threshold autoregression, limit cycles and cyclical data. In *Exploration Of A Nonlinear World: An Appreciation of Howell Tong’s Contributions to Statistics*, pages 9–56. World Scientific, 2009.
[4] Ruey S Tsay. Testing and modeling threshold autoregressive processes. *Journal of the American statistical association*, 84(405):231–240, 1989.
[5] Jesus Gonzalo and Michael Wolf. Subsampling inference in threshold autoregressive models. *Journal of Econometrics*, 127(2):201–224, 2005.
[6] Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, and Guangquan Zhang. Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12):2346–2363, 2018.
[7] Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM international conference on data mining*, pages 443–448. SIAM, 2007.
[8] Joao Gama, Pedro Medas, Gladys Castillo, and Pedro Rodrigues. Learning with drift detection. In *Brazilian symposium on artificial intelligence*, pages 286–295. Springer, 2004.
[9] Ewan S Page. Continuous inspection schemes. *Biometrika*, 41(1/2):100–115, 1954.
[10] Christoph Raab, Moritz Heusinger, and Frank-Michael Schleif. Reactive soft prototype computing for concept drift streams. *Neurocomputing*, 416:340–351, 2020.
[11] Joao Gama and Gladys Castillo. Learning with local drift detection. In *International conference on advanced data mining and applications*, pages 42–55. Springer, 2006.
[12] Manuel Baena-Garcia, José del Campo-Ávila, Raúl Fidalgo, Albert Bifet, R Gavalda, and Rafael Morales-Bueno. Early drift detection method. In *Fourth international workshop on knowledge discovery from data streams*, volume 6, pages 77–86, 2006.
[13] Isvani Frias-Blanco, José del Campo-Ávila, Gonzalo Ramos-Jimenez, Rafael Morales-Bueno, Agustin Ortiz-Diaz, and Yaile Caballero-Mota. Online and non-parametric drift detection methods based on hoeffding’s bounds. *IEEE Transactions on Knowledge and Data Engineering*, 27(3):810–823, 2014.
[14] Shuliang Xu and Junhong Wang. Dynamic extreme learning machine for data stream classification. *Neurocomputing*, 238:433–449, 2017.
[15] Myuu Myuu Wai Yan. Accurate detecting concept drift in evolving data streams. *ICT Express*, 6(4):332–338, 2020.
[16] Joao Gama, Raquel Sebastiao, and Pedro Pereira Rodrigues. On evaluating stream learning algorithms. *Machine learning*, 90(3):317–346, 2013.

&lt;page_number&gt;10&lt;/page_number&gt;

---


## Page 11

[17] Shuo Wang, Leandro L. Minku, Davide Ghezzi, Daniele Caltabiano, Peter Tino, and Xin Yao. Concept drift detection for online class imbalance learning. In *The 2013 International Joint Conference on Neural Networks (IJCNN)*, pages 1–10, 2013.

[18] Salvatore Greco and Tania Cerquitelli. Drift lens: Real-time unsupervised concept drift detection by evaluating per-label embedding distributions. In *2021 International Conference on Data Mining Workshops (ICDMW)*, pages 341–349, 2021.

[19] Bruce E Hansen. Inference in tar models. *Unpublished working paper. Chestnut Hill, MA: Boston College Department of Economics (December)*, 1996.

[20] Jacob Montiel, Jesse Read, Albert Bifet, and Talel Abdessalem. Scikit-multiflow: A multi-output streaming framework. *Journal of Machine Learning Research*, 19(72):1–5, 2018.

[21] Jerome H Friedman. Multivariate adaptive regression splines. *The annals of statistics*, pages 1–67, 1991.

[22] Rakesh Agrawal, Tomasz Imielinski, and Arun Swami. Database mining: A performance perspective. *IEEE transactions on knowledge and data engineering*, 5(6):914–925, 1993.

[23] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.

[24] Ernesto Aguilar Madrid and Nuno Antonio. Short-term electricity load forecasting with machine learning. *Information*, 12(2):50, 2021.

[25] Saverio De Vito, Ettore Massera, Marco Piga, Luca Martinotto, and Girolamo Di Francia. On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical*, 129(2):750–757, 2008.

[26] About the world air quality index project.

[27] Alexander Vergara, Shankar Vembu, Tuba Ayhan, Margaret A Ryan, Margie L Homer, and Ramón Huerta. Chemical gas sensor drift compensation using classifier ensembles. *Sensors and Actuators B: Chemical*, 166:320–329, 2012.

&lt;page_number&gt;11&lt;/page_number&gt;