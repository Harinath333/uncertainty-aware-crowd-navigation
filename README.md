# Uncertainity Aware Crowd Navigation
This repository contains the codes for our paper titled "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph" in ICRA 2023. 
Our extension focuses on uncertainty-aware crowd navigation by incorporating multi-hypothesis trajectory prediction, spatial uncertainty estimation, and risk-aware reinforcement learning to improve navigation performance in crowded environments.

## Abstract
Predicting pedestrian movement is an important part of safe robot navigation in crowded environments, since a person's future path may differ from a single predicted trajectory. This paper extends a Gumbel Social Transformer (GST)-based Proximal Policy Optimization (PPO) framework by retaining five stochastic trajectory hypotheses over a five-step prediction horizon. The spread of these predictions is measured using the trace of their spatial covariance. Intrusion risk is estimated separately, based on the proportion of hypotheses that enter a defined robot–human safety region, and is added as a penalty to the PPO reward. The existing GST, attention, GRU, and actor–critic architecture is retained. The extension was implemented in the PyTorch-based CrowdSim environment. In the reported ablation study, navigation success increased from 74.0% with the original GST-PPO configuration to 80.4% with multiple hypotheses alone and 84.0% when the risk penalty was also enabled. Collision and intrusion rates also decreased across these configurations. These results suggest that combining multiple predicted futures with risk-aware reward shaping may improve navigation performance in the tested setup. Further evaluation is needed to assess consistency across runs and determine how accurately the estimated intrusion risk reflects observed events.

<p align="center">
<img src="/figures/open.png" width="450" />
</p>

## Key Features

- **Multi-Hypothesis Prediction:** Retains five stochastic pedestrian trajectory hypotheses over a five-step prediction horizon.
- **Spatial Uncertainty Estimation:** Uses the trace of the spatial covariance matrix to measure the spread of predicted trajectories.
- **Intrusion-Risk Estimation:** Estimates risk from the proportion of predicted trajectories that enter the defined robot–human safety region.
- **Risk-Aware Reward Shaping:** Adds an intrusion-risk penalty to the existing prediction-aware PPO reward.
- **Ablation Study:** Compares the original GST-PPO baseline, the multi-hypothesis extension, and the complete risk-aware method.
- **Simulation-Based Evaluation:** Evaluates navigation performance using the PyTorch-based CrowdSim environment.

## Requirements

The project uses the following technologies:

- Python
- PyTorch
- CrowdSim simulation environment
- Gumbel Social Transformer (GST)
- Proximal Policy Optimization (PPO)
- NumPy and other project-specific dependencies

The exact package versions depend on the original repository and its configuration.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harinath333/uncertainty-aware-crowd-navigation.git
cd CrowdNav_Prediction_AttnGraph
```

Replace the placeholders with your repository URL and directory name.

### 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n uncrowdnav python=3.8
conda activate uncrowdnav
```

Use the Python version supported by your project's dependencies.

### 3. Install Dependencies

If the repository contains a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

If the project requires additional native libraries or compiled dependencies, install them according to the original CrowdSim/GST repository instructions.

## Run the Code

### Training

Before training, review the project configuration files.

Configure the following components according to the options supported by your implementation:

1. **Environment Configuration**
   - Select the CrowdSim environment.
   - Configure pedestrian behavior and simulation settings.
   - Set the prediction method and load the appropriate GST model.

2. **PPO Configuration**
   - Configure the PPO learning parameters.
   - Set the number of training episodes and parallel environments.
   - Configure the reward parameters, including the intrusion-risk penalty.

3. **Multi-Hypothesis Prediction**
   - Enable five stochastic trajectory hypotheses.
   - Set the prediction horizon to five steps.
   - Ensure the predicted positions are transformed into world coordinates before intrusion-risk evaluation.

4. **Risk-Aware Reward**
   - Configure the robot–human safety region.
   - Set the collision penalty.
   - Enable the intrusion-risk penalty when evaluating the complete method.

After configuring the project, run the training script provided by the repository. For example, if the entry point is `train.py`:

```bash
python train.py
```

**Note:** Confirm the actual script name and configuration options in your repository before running the command.

### Testing

Evaluate a trained model using the testing script and the appropriate model checkpoint.

For repositories using `test.py`, the command may be:

```bash
python test.py
```
Note that the `config.py` and `arguments.py` in the testing folder will be loaded, instead of those in the root directory.  
The testing results are logged in `trained_models/your_output_dir/test/` folder, and are also printed on terminal.  
If you set `visualize=True` in `test.py`, you will be able to see visualizations like this:  
<img src="/figures/visual.gif" width="420" />

Configure the test environment, checkpoint path, number of episodes, and visualization settings before running the evaluation.

Record the following metrics where supported:

- Success rate
- Collision rate
- Timeout rate
- Navigation time
- Path length
- Intrusion ratio
- Minimum intrusion distance
- Mean reward

Keep the evaluation conditions consistent when comparing different configurations.

## Experimental Configurations

The project evaluates three configurations.

| Configuration | Multi-Hypothesis Prediction | Risk Penalty |
|---------------|:---------------------------:|:------------:|
| A — Original GST-PPO | No | No |
| B — Hypotheses Only | Yes | No |
| C — Full Method | Yes | Yes |

This comparison helps examine the contribution of multi-hypothesis prediction and the additional risk penalty within the tested simulation setup.

## Results

The reported evaluation uses five rounds of 500 episodes per configuration, giving 2,500 episodes per configuration.

| Metric | Original GST-PPO | Hypotheses Only | Full Method |
|---|---:|---:|---:|
| Success Rate | 74.00% | 80.40% | 84.00% |
| Collision Rate | 23.52% | 18.20% | 16.00% |
| Timeout Rate | 2.48% | 1.40% | 0.00% |
| Navigation Time | 17.85 s | 15.65 s | 14.40 s |
| Path Length | 23.60 m | 21.80 m | 20.12 m |
| Intrusion Ratio | 14.20% | 12.10% | 9.53% |
| Minimum Intrusion Distance | 0.28 m | 0.33 m | 0.41 m |
| Mean Reward | 18.40 | 21.90 | 25.80 |

These are the reported aggregate results for the three configurations. Results may vary with model checkpoints, environment settings, and random seeds.

## Limitations

- Evaluation is currently limited to simulation.
- Five trajectory hypotheses may not represent every plausible pedestrian movement.
- Covariance trace measures spatial dispersion but does not establish calibrated predictive uncertainty.
- Intrusion risk is estimated using the robot's current position rather than its complete future trajectory.
- Further controlled experiments are needed to evaluate the contribution of individual components and assess generalization.

## Future Work

Potential directions include:

- Action-conditioned human–robot trajectory risk assessment.
- Improved calibration of spatial uncertainty and intrusion-risk estimates.
- Adaptive selection of trajectory hypotheses.
- Controlled comparisons under consistent experimental conditions.
- Evaluation on physical robotic platforms.

## References

The project builds on research in robot crowd navigation, trajectory prediction, graph-based interaction modeling, and reinforcement learning.

Please refer to the research paper and the reference list included in this repository for the complete bibliographic details.

## Contributors
Other contributors:  
[Mamidala Kishore Kumar](https://github.com/PeixinC)  
[Ambari Meena](https://github.com/PeixinC)  
[Veluru Sai Haritha](https://github.com/tedhuang96)   
[Udagandla Dinesh Kumar](https://github.com/TheNeeloy)

## Disclaimer

This project is intended for research and educational purposes. Performance observed in simulation does not guarantee equivalent performance in real-world environments. Real-world deployment requires additional testing, safety evaluation, and validation.

The reported results correspond to the specified experimental setup and should not be assumed to generalize to other environments without further evaluation.
