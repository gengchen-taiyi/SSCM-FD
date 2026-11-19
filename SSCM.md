**This is a link used to store the SSCM-FD method code.**



`Sample_data` is a demonstration data file and does not contain real operational data. 

Please install the required dependencies on your own; all are widely used, standard libraries for machine learning and deep learning.

**KAN Module Configuration:**

- The specific parameters of the KAN module can be adjusted empirically. Different parameter combinations may lead to substantial performance gains or degradation, which requires experimental tuning.
- Candidate symbolic function library: $x$, $tanh$, $log$, $exp$
- Numerical fitting uses 15 grid intervals with cubic B-splines.
- The network architecture should be defined according to the specific task. Note that based on our empirical findings, excessively deep or wide networks generally do not yield significant performance improvements.
- Adam is used as the default optimizer for KAN; alternative optimizers such as LBFGS are also applicable.
- The default regularization strength for numerical training is 0.0001. Other tunable hyperparameters include entropy penalty strength and pruning strength.
- The official reference implementation of KAN is the **pyKAN** library.

**File Structure & Usage:**

- Causal discovery can be executed via the `causal_discovery.py` script.
- We provide two implementations for the KAN module:
  - `KAN_1.py`: Version without symbolic processing. It features lower computational overhead and eliminates the need for time-consuming pruning operations.
  - `KAN_2.py`: Version with symbolic processing. Both implementations support manual adjustment of the network architecture and allow KAN experiments to be run in single or multiple stages.
- The core logic for fault detection and root-cause localization is implemented in `fault_diagnosis_core.py`
  - `fault_detection+root_location_1.py` corresponds to the original implementation presented in the manuscript.
  - `fault_detection+root_location_2.py` is the revised version with an added residual module, as mentioned in the response to reviewers, designed to address terminal sensor faults.

In practice, various auxiliary modules can be integrated based on the core script to further improve the algorithm's fault diagnosis performance. That said, an excessive number of modules may also introduce unnecessary complexity. The root-cause localization module is not affected by the addition or removal of these auxiliary modules.



Due to confidentiality agreements, we are unable to release any real industrial data from the top-blown furnace. We apologize for any inconvenience or ambiguity this may pose for reproduction.