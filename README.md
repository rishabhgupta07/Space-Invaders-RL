# Space Invaders RL: Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) agent to play the classic Atari game **Space Invaders**. It utilizes `Gymnasium` for the environment and `keras-rl2` for the reinforcement learning framework.


## 💡 Overview
The agent uses a Convolutional Neural Network (CNN) to "see" the game frames and learn optimal actions through an Epsilon-Greedy policy. Because `keras-rl2` is designed for the older OpenAI Gym, this project includes a custom **GymnasiumWrapper** to ensure compatibility with the latest `Gymnasium` (v0.29+) API.

## 🛠️ Tech Stack
* **Environment:** [Gymnasium](https://gymnasium.farama.org/) (Atari/ALE)
* **Framework:** [TensorFlow/Keras](https://www.tensorflow.org/)
* **RL Library:** `keras-rl2`
* **Logging:** Weights & Biases (WandB)

## 🚀 Getting Started

### Prerequisites
1. Install the Atari ROMs:
   ```bash
   pip install "gymnasium[atari,accept-rom-py]"
    ```


## 🚀 Installation & Setup

### 1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Space-Invaders-RL.git](https://github.com/YOUR_USERNAME/Space-Invaders-RL.git)
   cd Space-Invaders-RL
   ```


### 2. Install Dependencies: It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```



### 3. Final Push to Git
Now that you've added these files, run these commands in your terminal to update your public repository:

```bash
git add requirements.txt README.md
git commit -m "Add requirements and update setup instructions"
git push origin main
```