# Exploring Fairness in Predictive Process Monitoring
This repository contains the code, adapted methods, and experimental data for the research paper "Mind the Gap: Exploring Fairness in Predictive Process Monitoring".

This study empirically examines three bias mitigation strategies (pre-processing, in-processing, and post-processing) and evaluates their impact on both fairness and predictive performance of Predictive Process Monitoring (PPM) models. The experiments were conducted on twelve publicly available synthetic event logs with deliberately embedded historical bias, spanning four domains: hiring, hospital, lending, and renting.

The bias mitigation methods were adapted from the AI Fairness 360 (AIF360) library.

## Methodology 

### Bias Mitigation Strategies
We applied nine distinct methods from the AIF360 library, categorized as follows:

#### Pre-processing Methods
This approach aims to reduce historical bias by adjusting the training data's distribution before the model is trained.

- Reweighing: Adjusts the weights of training instances to ensure equal representation of protected and unprotected groups, correcting imbalances in the training data.

- Learning Fair Representations (LFR): Learns a latent representation of the dataset that encodes useful information while obfuscating details about the protected attribute.

- Disparate Impact Remover: Modifies the values of predictor attributes to reduce systematic statistical differences between groups. In this study, the full-repair option was adopted.

#### In-processing Methods
This approach incorporates fairness constraints directly into the model's training process and optimization function.

- Adversarial Debiasing: Employs a dual-model system where a predictor learns from the data, and an adversary attempts to detect bias in the predictor's outputs, penalizing it to encourage fairer representations.

- Prejudice Remover: Adds a discrimination-aware regularization term to the learning objective to mitigate bias during training.

- Meta Fair Classifier: Acts as a meta-engine that optimizes a classifier for a specified group fairness metric by enforcing a maximum allowable fairness violation (tau).

#### Post-processing Methods
This approach refines the outputs of a trained model to improve fairness without altering the underlying model itself.

- Equalized Odds: Modifies predictions to satisfy the equalized odds criterion, ensuring that false positive and false negative rates are similar between protected and unprotected groups.

- Calibrated Equalized Odds: Extends the Equalized Odds method by incorporating probabilistic calibration, randomizing certain predictions to achieve group-wise calibration and reduce error rate disparities.

- Reject Option Classification (ROC): Modifies predictions for individuals whose confidence scores lie within an uncertainty region around the decision boundary, preferentially assigning outcomes to balance fairness.

### Fairness Metrics
Fairness in the predictive models was measured using the following metrics:

- Demographic Parity (DP)

- Disparate Impact (DI)

- Equalized Odds (EO): Assessed through Equal Opportunity (EO-TPR) and Predictive Equality (EO-FPR).

## Repository Structure
The repository is organized as follows:

- FairnessProcessMining/: Contains the fairness methods from the AIF360 library that have been adapted for the process mining context.

- Hiring log/: Contains the event logs and experiment scripts for the hiring domain.

- Hospital log/: Contains the event logs and experiment scripts for the hospital domain.

- Lending log/: Contains the event logs and experiment scripts for the lending domain.

- Renting log/: Contains the event logs and experiment scripts for the renting domain.

- requirements.txt: A file listing all the Python dependencies required to run the experiments.
