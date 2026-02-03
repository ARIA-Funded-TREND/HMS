
# Reinforcement Learning

This repository contains the implementation of Reinforcement Learning (RL) algorithm for **Scalable Machines with Intrinsic Higher Mental States**, integrated with Google’s PI framework ([Brain Tokyo Workshop](https://github.com/google/brain-tokyo-workshop)).

The implementation utilizes **CO4** and is designed to be highly modular, with individual classes handling specific logic to maintain scalability despite the increased code length.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.x**
* **CPU-Heavy Environment**: Rollouts run on the CPU; GPU acceleration is currently not utilized for this specific implementation.

### Installation

1. **Clone and Navigate**
```bash
cd AttentionNeuron

```

2. **Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements_comprehensive.txt

```



---

## 🛠 Usage

### Training an Agent

To begin training, use the `train_agent.py` script with the desired configuration file.

```bash
python train_agent.py --config configs/x.gin --log-dir log/x --num-workers 6

```

* `--config`: Path to your `.gin` configuration.
* `--log-dir`: Directory where logs and model checkpoints will be saved.
* `--num-workers`: Number of parallel workers for rollouts.

### Evaluation

To evaluate a trained model and check its performance:

```bash
python eval_agent.py --log-dir log/x --model-filename best.npz

```

---


## ⚙️ Configuration

Each task has specific parameters in its corresponding `config.gin` file. Below are the key tunable settings:

### Common to All Tasks

* `shuffle_on_reset` or `permute_obs`: Defines whether to shuffle observations during rollouts.
* `render`: Toggle environment visualization.
* `v`: Verbosity toggle for logging.

### Task-Specific Tuning

| Task | Parameter | Description |
| --- | --- | --- |
| **CartPoleSwingUp** | `num_noise_channels` | Defines the number of noisy input channels to the agent. |
| **CarRacing** | `bkg` | Set background image (e.g., `bkg="mt_fuji"` loads `tasks/bkg/mt_fuji.jpg`). Use `None` for default. |
| **CarRacing** | `patch_size` | Defines the size of the patch to be shuffled. |

---

## 📊 Training Results: Google vs. Co4

Below is a comparison of training progress across various RL environments, showcasing the performance of the **Co4** implementation against the baseline.


| **Environment** | **Acrobot** | **CarRacing** |
| :--- | :---: | :---: |
| **Results** | ![Acrobot](images/AcrobotGooglevsCMI.png) | ![CarRacing](images/CarRacingGooglevsCMI.png) |

| **Environment** | **CartPole** | **MountainCar** |
| :--- | :---: | :---: |
| **Results** | ![CartPole](images/CartPoleGooglevsCMI.png) | ![MountainCar](images/MountainCarGooglevsCMI.png) |

| **Environment** | **PyAnt** |
| :--- | :---: |
| **Results** | ![PyAnt](images/PyAntGooglevsCMI.png) |

---

## 📂 Project Structure

* `AttentionNeuron/`: Core logic and agent implementation.
* `configs/`: `.gin` files for experiment hyperparameters.
* `log/`: Training logs and saved `.npz` model weights.
* `images/`: Visualization plots for documentation.
