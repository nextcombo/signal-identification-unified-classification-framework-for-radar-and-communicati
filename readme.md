# [Signal Identification]Unified Classification Framework for Radar and Communication Signals Based on LSTM, with SOTA Performance and More Simplified Structure[Attached Python Code]

In the field of radar and communication signal processing, signal modulation classification is a critical link in both civilian and military receivers. Traditional classification methods either have high computational complexity or rely excessively on expert knowledge. The advent of deep learning has provided a new solution to this issue. Previous research on radar and communication signal classification was often conducted separately, but in reality, they share similar characteristics and methods can be mutually referenced.
Recently, a research team from the University of Madrid, Spain, and the University of Illinois at Urbana-Champaign proposed a simple yet unified approach for radar and communication signal classification based on Long Short-Term Memory (LSTM) neural networks. This not only achieved SOTA performance in radar signal classification but also publicly released the DeepRadar2022 standardized radar dataset, providing a unified benchmark for method comparison in the field.
1. Signal Modulation Classification: Limitations of Traditional Methods and New Opportunities for Deep Learning
Automatic Modulation Classification (AMC) is a key step after signal detection that automatically determines the modulation method of the collected samples, serving as the premise for data demodulation and is widely used in various types of receivers. Traditional AMC algorithms are mainly divided into two categories:
1. Likelihood Basis (LB) algorithm: It achieves an optimal solution by minimizing the error classification probability, but it requires the probability density function to contain all information about the waveform, which results in extremely high computational complexity and difficulty in adapting to real-world work environments;
2. Feature Basis (FB) algorithm: It extracts representative features of the signal for classification, is easy to implement and suitable for real-time applications, but highly dependent on expert knowledge, with a long development cycle and relatively poor performance in specific environments.
The emergence of machine learning (ML) and deep learning (DL) has broken through the limitations of traditional methods, enabling high-performance and high-precision AMC without the need for time-consuming feature engineering. Current algorithms such as K-nearest neighbors, support vector machines, multilayer perceptrons, recurrent neural networks (RNN), convolutional neural networks (CNN), etc., have been applied in this field. The outstanding performance of LSTM networks in sequence data processing and classification has led the research team to see its potential in unifying radar and communication signal classification.
2. Current Research Status: Disjointed Development of Radar and Communication Signal Classification Despite ML/DL being used for modulation classification of radar and communication signals, their research has long been disjointed, with different approaches focusing on each domain:
Communication signals: The most typical representation is a time series, where one-dimensional CNNs are the most widely used classification methods. Models like GGCNN and SE-MSFN have achieved excellent results based on public communication dataset RadioML 2018.01A;
Radar signals: Traditional methods often rely on feature extraction combined with MLPs, while recent years have often transformed signals into time-frequency graphs, paired with two-dimensional CNNs (and various improved versions, such as incorporating reinforcement learning and denoising autoencoders) or pre-trained CNN models (AlexNet, Inception), with a few studies using RNNs for time-series processing of radar signals, such as attention-based LSTMs and GRUs.
This disjointed research model has not fully utilized the similarities between the two, nor has it formed standardized comparisons due to the lack of publicly available radar signal datasets.
3. Experimental Foundation: Signal Model and Three Core Data Sets To achieve unified classification of radar and communication signals, the research team has defined a unified signal model and selected/constructed three types of uniformly regulated datasets to ensure consistency in input for the LSTM framework.
3.1 In the experiments involving the unified signal model, all radar and communication signals follow the following model:
The product is a normalized signal with unit power, where A is the signal power scaling factor and represents additive Gaussian noise (with real and imaginary parts variances). The corresponding Signal-to-Noise Ratio (SNR) is:
All signals are of size 1024×2, corresponding to the in-phase and quadrature components of time samples, meeting the fixed input size requirements of the LSTM framework.
3.2 Three major experimental datasets (1) Communication signal dataset: RadioML 2018.01A This public dataset contains 2555904 signals, with an SNR range from -20dB to 30dB (step size of 2dB), covering 24 types of digital and analog modulation methods (such as OOK, ASK, PSK, QAM, AM, FM, GMSK, etc.), including synthetic channel signals generated by GNU Radio and airborne (OTA) recorded signals. The only limitation is that the signal generation parameters and type (synthetic/OTA) are not explicitly stated.
(2) Radar signal dataset 1: DeepRadar2022 (team self-developed public) This is a dataset built to fill the gap in radar signal public datasets by the research team, containing 23 categories of continuous and pulsed radar signals, specifying key parameters for each category such as sampling frequency, carrier frequency, and pulse width (sampling frequency fixed at 100MHz, input sequence fixed at 1024 samples, short-duration signals achieved through interpolation for uniform size).
This dataset consists of 782000 signals, including 469200 training sets, 156400 validation sets, and 156400 test sets, which have been publicly available on Kaggle and provide a standardized benchmark for radar signal classification.
(3) Radar signal dataset 2: Eight-class replica dataset To compare with existing SOTA methods for radar signal classification, the research team replicated the eight-class radar signal dataset proposed by SABLNet, containing CW, LFM, BFSK, etc., with an SNR range from -20dB to 20dB (step size of 2dB), supporting both raw time series and autocorrelation preprocessing as input formats.
IV. Core Solution: Three-Layer Stacked LSTM Unified Classification Framework
4.2 Framework Structure Design: The LSTM framework proposed by the research team is designed to be simple and efficient. It takes signal time series as input, with a highly simplified structure consisting only of three stacked LSTM layers and a fully connected classification layer:
1  
LSTM layer: Each layer contains 128 cells, the first two layers return all sequence values (1024×128), and the last layer only returns a single value of the entire sequence (1×128);
2  
Category layer: Softmax activation function is used, and the number of output neurons adjusts with the number of categories in the dataset (1032/2967/3096 parameters correspond to 8/23/24 classes respectively).
The entire framework is implemented based on TensorFlow 2.4.1, with only 330240 trainable parameters in addition to the classification layer, significantly lower than existing complex models.
4.3 Training hyperparameters and evaluation metrics: Batch size of 256, employing the Adam optimizer with cyclical learning rates (1×10⁻⁷ to 1×10⁻³ oscillations) to avoid converging to local optima, training for 300 epochs from scratch.
Evaluation indicators: ① The average classification accuracy of all categories with SNR changes; ② The minimum SNR (sensitivity) required to achieve a classification accuracy of 90%; ③ Confusion matrices under different SNRs.
5. Experimental Results: Full-Scene SOTA, Structure Simplification Still Performs Advantageously
The research team has validated the LSTM framework's classification performance across eight categories of radar signals and twenty-three categories (DeepRadar2022) through multiple experiments. Compared to existing SOTA methods, it demonstrated superior accuracy, sensitivity, and robustness.
5.1 Layer Verification: Three-Layer LSTM Optimal Structure
Based on the simplest eight-category radar dataset, the research team compared the performance of one-layer, two-layer, and three-layer LSTMs. The results are shown in Figure 3, where a one-layer LSTM significantly lags behind both two-layer and three-layer stacked LSTMs. Therefore, all subsequent experiments adopted this structure.
5.2 Eight-Category Radar Signals: Comparable to SABLNet, No Preprocessing Needed for Greater Simplicity
By comparing the LSTM framework with the SOTA model SABLNet (a CNN+Bidirectional LSTM+Attention Mechanism, which is more complex), two key conclusions were drawn:
1. Input Domain Influence: After applying autocorrelation preprocessing to the sequence, there was no significant performance improvement. The LSTM framework can directly learn features from raw time series without the need for additional preprocessing, outperforming SABLNet that relies on preprocessing[Figures 4/5];
2. Robustness Exceptionally Good: When the training set SNR range was expanded from -12:2:20dB to -20:2:20dB, the framework maintained a classification accuracy of 90% under high SNR scenarios, achieving a minimum SNR of -10dB, consistent with SABLNet[Figure 4].
5.3 Twenty-Three-Category Radar Signals (DeepRadar2022): Sensitivity Slightly Decreases Due to Increased Categories, Higher Order Phase Modulation Being the Main Challenge
As the number of categories increased from eight to twenty-three, the complexity of the dataset significantly increased. Consequently, the minimum SNR required for the framework to achieve a 90% accuracy dropped from -10dB to **-2dB**[Figure 6], which is a reasonable performance change due to the increase in category numbers.
Experimental findings reveal that high-order phase modulation signals are the primary classification challenges: 4PSK and 8PSK can easily be confused with each other, and Frank code, P1-P4 codes also face similar issues with their phase modulation signals [Figures 7/8]. The research team proposes to initially categorize them as PSK before further distinguishing by order to address this issue.
5.4 24-Class Communication Signals: Outperforms GGCNN and is comparable to SE-MSFN in performance, with a more streamlined structure and superior robustness based on the RadioML 2018.01A dataset. A comparison of the LSTM framework with SOTA models GGCNN and SE-MSFN for communication signal classification:
1  
Overall Performance: The framework achieves a minimum SNR of 6dB (due to the difficulty in distinguishing high-order PSK/QAM, AM-DSB/SSB) to reach a 90% accuracy rate. It outperforms GGCNN and is on par with SE-MSFN, but the framework does not require feature extraction steps, has fewer layers, and is simpler in structure [Figure 9];
2  
Single Signal Sensitivity: Compared with the original author's synthesis signal classification results from RadioML, signals such as 64QAM and 256QAM in the original method could not achieve a 90% accuracy rate even at a maximum evaluation SNR of 20dB, whereas all signals except AM-SSB-SC in the LSTM framework can achieve a 90% accuracy rate when SNR is below 10dB, demonstrating superior robustness [Figure 10].
Sixth, Research Conclusions: The research team's proposed unified classification framework based on three-layer stacked LSTM has achieved integrated modulation classification for radar and communication signals for the first time. The core conclusions are as follows:
1  
Performance at SOTA: Achieves the best performance among existing deep learning methods on both 8-class/23-class radar signals and 24-class communication signals, on par or better than complex models (SABLNet, SE-MSFN);
2  
Structure is simpler: Without additional feature extraction or preprocessing steps, it can train parameters far fewer than existing models, making it easy to implement and deploy;
Higher sensitivity and robustness: Can directly learn features from raw time series, exhibits good robustness in SNR range, achieving 90% classification accuracy for most signals at lower SNR;
4. Filling the dataset gap: The publicly available DeepRadar2022 dataset contains 23 radar signal categories, providing an important benchmark for standardized methods of radar signal classification.
This research offers new ideas for unified processing of radar and communication signals, with the simplified LSTM framework demonstrating higher practical application value in real-world engineering. Additionally, optimization for classifying high-order phase modulated signals will become a focus of future research.

## Images

![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](01.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](02.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](03.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](04.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](05.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](06.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](07.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](08.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](09.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](10.jpg)
![(Signal Identification)A unified classification framework for radar and communication signals based on LSTM, with SOTA performance and more concise structure. [Attached Python code]](11.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
