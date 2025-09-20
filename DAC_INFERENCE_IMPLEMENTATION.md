# DAC Inference and Behavior Plotting Implementation

## Summary

Successfully implemented periodic inference and behavior plotting functionality for DAC, making it consistent with the option-critic implementation. The DAC training now includes:

1. **Periodic inference** every few episodes (configurable)
2. **Training behavior plotting** after each episode
3. **Best testing behavior plotting** after each inference round

## Files Modified/Created

### 1. New Files Created

#### `RL/dac/dac_evaluation.py`
- **Purpose**: Evaluation function for DAC model with behavior data collection
- **Key Features**:
  - Runs multiple evaluation episodes with deterministic policy
  - Collects action sequences, option sequences, and option instances
  - Calculates success rate and performance metrics
  - Returns evaluation history for visualization
- **Function**: `evaluate_dac_model(agent, env_wrapper, num_episodes, logger, seed)`

#### `test_dac_inference.py`
- **Purpose**: Test script to verify the new DAC functionality
- **Features**: Tests periodic inference, training plotting, and test plotting

### 2. Files Modified

#### `RL/dac/dac_config.py`
- **Added Parameters**:
  ```python
  eval_frequency: int = 10    # Periodic inference frequency (episodes)
  eval_episodes: int = 5      # Number of episodes per evaluation
  ```

#### `RL/dac/dac_trainer.py`
- **Major Additions**:
  - **Behavior tracking dictionaries** (like option_critic):
    - `training_histories`: Tracks all training episode data
    - `evaluation_histories`: Tracks best evaluation episode data

  - **Modified training loop**:
    - Added periodic inference every `eval_frequency` episodes
    - Added training behavior plotting after each episode
    - Added test behavior plotting after inference rounds

  - **New Methods**:
    - `evaluate_and_save(episode_num)`: Periodic evaluation with visualization
    - `plot_training_behaviors(episode_num)`: Training behavior visualization
    - `plot_test_behaviors(episode_num, eval_history)`: Test behavior visualization

  - **Enhanced episode tracking**:
    - Collects action sequences, option sequences, and option instances
    - Stores behavior data for visualization

## Key Features Implemented

### 1. Periodic Inference (像 option_critic 一樣)
- **Frequency**: Configurable via `config.eval_frequency` (default: every 10 episodes)
- **Episodes per round**: Configurable via `config.eval_episodes` (default: 5 episodes)
- **Functionality**:
  - Runs deterministic evaluation episodes
  - Collects behavior data for the best episode
  - Saves best models automatically
  - Generates test behavior visualizations

### 2. Training Behavior Plotting (每個episode結束後)
- **Frequency**: After every training episode
- **Output**: `training_behaviors_ep{N}.png` files
- **Content**: Shows action sequences and option usage during training
- **Uses**: Same visualization function as option_critic (`plot_test_behaviors`)

### 3. Testing Behavior Plotting (inference結束後最佳episode)
- **Frequency**: After each inference round
- **Output**: `test_behaviors_inference_ep{N}.png` files
- **Content**: Shows the best episode from each evaluation round
- **Tracks**: Accumulated best episodes across all inference rounds

### 4. Data Collection Compatibility
- **Action sequences**: Compatible with existing visualization
- **Option sequences**: Tracks option usage across episodes
- **Option instances**: Detailed option termination tracking
- **SCWB compatibility**: Maintains compatibility with existing plotting

## Usage

### Basic Usage
```python
from RL.dac.dac_config import DACConfig
from RL.dac.dac_trainer import DACTrainer

# Create config with inference parameters
config = DACConfig(
    eval_frequency=10,    # Inference every 10 episodes
    eval_episodes=5,      # 5 episodes per inference round
    max_episodes=100
)

# Create and run trainer
trainer = DACTrainer(config, base_environment, logger)
results = trainer.train()
```

### Generated Files
After training, you'll find in the log directory:
- `training_behaviors_ep1.png`, `training_behaviors_ep2.png`, ... (every episode)
- `test_behaviors_inference_ep10.png`, `test_behaviors_inference_ep20.png`, ... (every eval_frequency episodes)

## Compatibility

### With option_critic
- Uses the same `plot_test_behaviors` function
- Same data structure for behavior tracking
- Compatible Record-like objects for visualization

### With existing DAC code
- Maintains all existing DAC functionality
- Backward compatible with existing training scripts
- Optional features (can be disabled by setting high eval_frequency)

## Testing

Run the test script to verify functionality:
```bash
cd /mnt/d/kyle_MD_project/optionRL/GraphRL_story_level
python test_dac_inference.py
```

The test creates a small training run and verifies:
- Periodic inference works
- Behavior plots are generated
- Data collection is working correctly

## Notes

- The implementation closely follows the option_critic pattern for consistency
- Uses the existing `plot_test_behaviors` function to ensure visual compatibility
- Maintains the same behavior tracking data structures
- All new functionality is configurable and can be disabled if needed
- Performance impact is minimal as plotting happens after episodes complete

## Fixes Applied

### 1. Training Mode Handling
**Issue**: `'DACAgent' object has no attribute 'training'`
**Fix**: Use `agent.network.training` instead of `agent.training` in evaluation function

### 2. Environment Logger
**Issue**: Missing `logger` parameter in Environment constructor
**Fix**: Added `logger=logger` parameter in test script

### 3. Network Dimensions
**Issue**: Shape mismatch in neural network (`mat1 and mat2 shapes cannot be multiplied`)
**Fix**: Updated config to use correct dimensions:
- `node_feature_dim=8` (actual graph node features)
- `edge_feature_dim=13` (actual graph edge features)

### 4. Environment Wrapper Access
**Issue**: Plotting functions need access to base environment
**Fix**: Use `getattr(self.env, 'base_env', self.env)` to safely access base environment

## Implementation Status: ✅ COMPLETE

The DAC implementation now successfully includes:
- ✅ Periodic inference every N episodes
- ✅ Training behavior plotting after each episode
- ✅ Test behavior plotting after inference rounds
- ✅ Compatible with existing option_critic visualization functions
- ✅ Full behavior data collection and tracking