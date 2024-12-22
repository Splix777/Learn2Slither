<h1 align="center">Learn2Slither</h1>
<h2 align="center">Reinforcement Learning</h2>

<div align="center">
<img src="resources/readme/logo.png" alt="Neural Brain" width="25%">
</div>

# Table of Contents

- [A Machine Learning Project](#a-machine-learning-project)
- [Environment and Board](#environment-and-board)
  - [Board Configuration](#board-configuration)
  - [Game Logic](#game-logic)
    - [Resetting the Environment](#resetting-the-environment)
    - [Map Management](#map-management)
    - [Collision and Events Handling](#collision-and-events-handling)
  - [Game Steps](#game-steps)
  - [State Representation](#state-representation)
- [Snake Class](#snake-class)
  - [Overview](#overview)
  - [Key Attributes](#key-attributes)
  - [Main Methods](#main-methods)
  - [Gameplay Logic](#gameplay-logic)
- [The Snake's Brain: Agent Class](#the-snakes-brain-agent-class)
  - [Overview](#overview-1)
  - [Model, Optimizer, and Loss](#model-optimizer-and-loss)
    - [Model Architecture](#model-architecture)
    - [Optimizer and Loss Function](#optimizer-and-loss-function)
  - [Hyperparameters](#hyperparameters)
  - [Training Process](#training-process)
  - [Training Step (`train_step`)](#training-step-train_step)
    - [Implementation](#implementation)
  - [Experience Replay (`replay`)](#experience-replay-replay)
  - [Action Selection (`act` and `choose_action`)](#action-selection-act-and-choose_action)
  - [Connection Between Snake, Environment, and Agent](#connection-between-snake-environment-and-agent)


# A Machine Learning Project

This project is an artificial intelligence project about reinforcement learning.

Reinforcement Learning (RL) stands at the forefront of artificial intelligence, rep-
resenting a paradigm where intelligent agents learn optimal decision-making strategies
through interaction with their environment. Unlike traditional programming approaches,
RL enables machines to learn by trial and error, continuously refining their behavior
based on feedback in the form of rewards or punishments.

In this project, we create a snake, moving on a board, and controlled by an
intelligent agent. The board and the agent follow specific rules and constraints that are
detailed below. I encourage you to read the subject located in the docs directory to familiarize yourself with the project rules and requirements.

## Environment and Board

The environment of the snake game is encapsulated in the `Environment` class found in [src/game/environment.py](src/game/environment.py). Here's a breakdown of its key functionalities:

### Board Configuration

- **Board Size**: Fixed at 10x10.
- **Apples**: 
  - **Green Apples**: Two are placed randomly on the board. When eaten, they increase the snake's length by 1, and a new green apple appears.
  - **Red Apple**: One is placed randomly. Eating it decreases the snake's length by 1, with a new red apple appearing after consumption.
- **Snake**: Starts with a length of 3 cells, placed randomly but contiguously.
- **Game Over Conditions**: 
  - Hitting walls.
  - Self-collision or collision with another snake.
  - Snake's length drops to 0.

### Game Logic

#### Resetting the Environment:
- The `reset()` method initializes the board, placing snakes and apples, ensuring each snake has enough space to start.

#### Map Management:
- **Map Creation**: A blank map with walls is initialized, then populated with snakes and apples.
- **Apple Distribution**: Apples are strategically placed in empty cells without overlapping snakes or other apples.

#### Collision and Events Handling:
- **Collision Checks**: The environment checks if a snake has collided with a wall, itself, or another snake, ending the game if true.
- **Apple Consumption**: Detects if a snake has eaten an apple, adjusting the snake's length accordingly and updating apple positions.

### Game Steps

- **Train Step (`train_step`)**: 
  - Used during training phases. Each snake moves based on a learned policy, and the environment records states, actions, rewards, and whether the snake died for later learning updates.
  - Utilizes a buffer to store game experiences which are then cached in the snake's brain for reinforcement learning.

- **Regular Step (`step`)**: 
  - Moves all snakes based on their current direction or a given action, updates collisions and apple consumption, but does not involve learning or saving experiences.

### State Representation

The state of the environment for each snake is crucial for decision-making:

- **State Components**:
  - **Direction**: One-hot encoding of the current direction the snake is facing.
  - **Options**: Encoded possible moves based on the snake's current position.
  - **Nearby Risks**: A vector indicating immediate dangers in four cardinal directions (walls, other snakes).
  - **Apple in Sight**: Detects if there's an apple (green or red) in each direction, with positive values for green and negative for red apples.
  - **Surroundings**: A detailed view of obstacles, green apples, and red apples within a certain radius in each direction.

- **Why States Matter**: 
  - **Decision Making**: States provide the sensory input for the snake's AI to decide its next move.
  - **Learning**: For training, states help in associating actions with outcomes, forming the basis for reinforcement learning algorithms to improve gameplay over time.

- **State Collection**: 
  - The `get_state` method compiles all this information into a tensor for neural network processing, normalizing distances and encoding categorical information into numerical formats suitable for machine learning.

This environment setup allows for a dynamic and challenging game where strategic thinking, based on the complex state information, can lead to better survival rates and higher scores. The code provides a robust foundation for further AI development or for human players to interact with through an interface.

## Snake Class

The `Snake` class is central to gameplay and can be found in [src/game/snake.py](src/game/snake.py). Below is an in-depth look at its structure and functionality:

### Overview

- **Initialization**: Each snake has an ID, configuration settings, and an optional AI brain which decides its movements during training or gameplay.

### Key Attributes

- **State Tracking**:
  - `alive`: Boolean to indicate if the snake is still in play.
  - `size`: Current length of the snake.
  - `head` and `body`: Coordinates of the snake's head and body segments.
  - `movement_direction`: Current direction of movement.

- **Counters and Rewards**:
  - `kills`, `red_apples_eaten`, `green_apples_eaten`: Stats for game metrics.
  - `reward`: Used for reinforcement learning, updated based on actions and outcomes.
  - `steps_without_food`: Counts moves without eating to determine if the snake is starving.

### Main Methods

- **Initialization (`initialize`)**:
  - Sets up the snake at a start position with an initial direction, creating its body in line with this direction.

- **Movement (`move`)**:
  - Moves the snake one step in its current direction. 
  - If an AI brain is present, it uses the state of the environment to decide the next move. 
  - Handles the logic for body growth or shrinkage, ensuring the snake's body length matches its size.

- **Direction Control (`snake_controller`)**:
  - Allows changing the snake's direction based on input, ensuring the snake can't immediately reverse direction.

- **Eating Mechanics**:
  - `eat_green_apple`: Increases snake size by one and resets food steps counter.
  - `eat_red_apple`: Decreases snake size, potentially leading to death if size becomes zero.

- **Death and Starvation**:
  - `death`: Ends the snake's life, resetting its attributes to default values.
  - `starving`: Checks if the snake has moved too many steps without food, implementing a penalty if true.

- **State Representation**:
  - `one_hot_direction`: Provides a one-hot encoding of the snake's current direction.
  - `one_hot_options`: Encodes possible movements for the snake, excluding the opposite of its current direction.

### Gameplay Logic

- **Movement Strategy**: When a snake moves, its body shifts forward, with the new head position calculated based on the current direction. If the snake's size is less than its body length, the last segment is removed, creating a growth effect when eating green apples or a shrinkage when consuming red ones.

- **Collision Handling**: Not explicitly in the snake class but interacts with the environment for collision detection leading to death or interaction with apples.

This class encapsulates the behavior of the snake in the game, providing all necessary methods for movement, interaction with the environment, and managing its own state. It's designed to work seamlessly with an AI agent for autonomous play or direct user control for manual gameplay.

## The Snake's Brain: Agent Class

The intelligence behind the snake's movement lies in the `Agent` class, located at [src/ai/agent.py](src/ai/agent.py). This class implements a neural network for decision-making, using reinforcement learning principles.

### Overview

- **Integration with Snake**: Each snake instance can have an `Agent` as its brain, which decides the snake's action based on the current game state. The snake's `move` method uses this brain to select the next direction.

- **State and Action**: The agent takes the game state (encoded as a tensor) as input and outputs Q-values for each possible action (direction the snake can move).

### Model, Optimizer, and Loss

#### Model Architecture
- The neural network is structured with:
  - An input layer matching the state size.
  - A hidden layer with 128 neurons and LeakyReLU activation for non-linearity.
  - An output layer with neurons equal to the number of actions.

#### Optimizer and Loss Function
- **Optimizer**: Adam is used for optimization, with AMSGrad and weight decay for regularization, helping in achieving faster convergence and preventing overfitting.
- **Loss Function**: Smooth L1 Loss (Huber Loss) is employed to handle outliers in Q-value predictions, balancing between mean squared error and mean absolute error.

### Hyperparameters

- **Gamma (`gamma`)**: Discount factor for future rewards in the Bellman equation, setting how much future rewards are valued compared to immediate rewards.
- **Batch Size (`batch_size`)**: Number of experiences processed in one training iteration, balancing between computational efficiency and learning from diverse experiences.
- **Epochs (`epochs`)**: Number of complete passes through the dataset during each training session.
- **Epsilon (`epsilon`)**: Exploration rate for epsilon-greedy policy, determining how often to take random actions for exploration.
- **Decay Rate (`decay`)**: Rate at which `epsilon` decreases over time, shifting from exploration to exploitation.
- **Minimum Epsilon (`epsilon_min`)**: Lowest value `epsilon` can decay to, ensuring some level of exploration persists.
- **Learning Rate**: Controls how much to adjust the model's weights with respect to the loss gradient.

### Training Process

### Training Step (`train_step`)

**Bellman Equation for Learning**: The core of Q-learning involves:

- Calculating the current Q-values for the given state.
- Predicting future Q-values with the `gamma` discount factor.
- Updating the Q-value for the chosen action using the formula:

  $$ Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right) $$

  Where:

  - $ Q(s, a) $: Current Q-value for state $ s $ and action $ a $.
  - $ \alpha $: Learning rate.
  - $ r $: Reward from taking action $ a $.
  - $ \gamma $: Discount factor for future rewards.
  - $ \max_{a'} Q(s', a') $: Maximum Q-value for the next state $ s' $ across all possible actions $ a' $.
  - $ s, s' $: Current and next states.
  - $ a, a' $: Current and next actions.

- **Batch Processing**: Experiences are batched to train the model, allowing for multiple updates based on different game scenarios in one step.

### Implementation

In the code, the Bellman equation is implemented as:

```python
target_q_values = reward + (1 - done) * self.gamma * max_next_q_values
```
- **Variable Explanation**:
  - **reward**: This variable corresponds to $ r $ in the Bellman equation. It's the immediate reward received after taking action a in states. In our code, we use torch.clamp to ensure that the reward doesn't go beyond a certain range, typically between -1 and 1, to normalize it for consistent learning.
  - **done**: This boolean flag indicates whether the episode has ended. If done is True, it means no further steps can be taken, so future rewards are irrelevant. In the equation, (1 - done) is used to nullify the future reward term when the episode terminates, effectively making it:
    - `1 - done = 1` if the episode continues, allowing future rewards to be considered.
    - `1 - done = 0` if the episode ends, disregarding future rewards.
  - **self.gamma**: This is the discount factor `γ` from the Bellman equation. It determines how much future rewards are valued compared to immediate rewards. A value closer to 1 means future rewards are more significant, while a value closer to 0 emphasizes immediate rewards.
  - **max_next_q_values**: This corresponds to $ \max_{a'} Q(s', a') $. It's the maximum Q-value of the next state over all possible actions `(s', a')`. This value is calculated by taking the neural network's predictions for the next state and finding the highest Q-value among all actions for that statet.

#### Experience Replay (`replay`)

- The agent stores experiences in a memory buffer (`deque`) and samples from this to break correlation in the data, leading to more stable and less noisy gradients during training.

#### Action Selection (`act` and `choose_action`)

- **Epsilon-Greedy Strategy**: Balances exploration (random action) and exploitation (best known action) based on `epsilon`. During training, `epsilon` starts high and decays, encouraging initial exploration and later exploitation of learned strategies.

### Connection Between Snake, Environment, and Agent

- **Snake**: Collects states, executes actions based on the agent's decisions, and manages its own reward based on game events (eating apples, collisions).
- **Environment**: Provides the game state, handles game rules, and updates the game world according to snake actions.
- **Agent**: Learns from these interactions by updating its policy through the reinforcement learning process, aiming to maximize the cumulative reward over time.

This setup creates a feedback loop where the snake's performance in the environment informs the agent's learning process, allowing for dynamic strategy adaptation. The agent's role is pivotal, as it's where all the AI learning logic resides, making decisions that directly impact the snake's survival and success in the game environment.