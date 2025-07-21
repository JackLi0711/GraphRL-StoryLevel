import torch
import typing
import collections
import numpy as np


_field_names = [
    "graph",
    "action",
    "reward",
    "next_graph",
    "done",
    "infeasible_actions", 
    "aux"
]


Experience = collections.namedtuple("Experience", field_names=_field_names)


class ExperienceReplayBuffer:
    """Fixed-size buffer to store Experience tuples."""
    def __init__(self,
                 batch_size: int,
                 buffer_size: int = None,
                 random_state: np.random.RandomState = None,
                 logger = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        random_state (np.random.RandomState): random number generator.

        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer = collections.deque(maxlen=buffer_size)
        self._random_state = np.random.RandomState() if random_state is None else random_state
        self._logger = logger


    def __len__(self) -> int:
        return len(self._buffer)
    

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size
    

    @property
    def buffer_size(self) -> int:
        """Total number of experience samples stored in memory."""
        return self._buffer_size
    

    def append(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        self._buffer.append(experience)
        self._logger.info(f"buffer len: {self.__len__()}")
        # print(f"buffer len: {self.__len__()}, memory allocated: {torch.cuda.memory_allocated()/1e+06:.2f} MB")
        

    def sample(self) -> typing.List[Experience]:
        """Randomly sample a batch of experiences from memory."""
        idxs = self._random_state.randint(len(self._buffer), size=self._batch_size)
        experiences = [self._buffer[idx] for idx in idxs]
        return experiences




class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""
    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 prioritized_alpha: float = 0.0,
                 random_state: np.random.RandomState = None,
                 logger = None) -> None:
        """
        Initialize a PrioritizedExperienceReplayBuffer object.

        Parameters:
        -------------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        prioritized_alpha (float): strength of prioritized sampling, default to 0.0 (i.e., uniform sampling)
        random_state (np.random.RandomState): random number generator

        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0     # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._prioritized_alpha = prioritized_alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        self._logger = logger


    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    
    @property
    def prioritized_alpha(self) -> float:
        """Strength of prioritized sampling."""
        return self._prioritized_alpha


    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size

    
    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size


    def append(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():  # replace the lowest priority experience with the new experience
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass    # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1
        self._logger.info(f"buffer len: {self.__len__()}")


    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0


    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size


    def sample(self, bias_correcting_beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of experience from memory."""
        # use sampling scheme to determine which experiences to use for learning
        priorities = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = (priorities ** self._prioritized_alpha) / np.sum(priorities ** self._prioritized_alpha)
        sampled_idxs = self._random_state.choice(np.arange(priorities.size),
                                                 size=self._batch_size,
                                                 replace=True,
                                                 p=sampling_probs)

        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][sampled_idxs]
        weights = (self._buffer_length * sampling_probs[sampled_idxs]) ** (-1 * bias_correcting_beta)
        normalized_weights = weights / weights.max()

        return sampled_idxs, experiences, normalized_weights


    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities
        print(f"mean priority: {np.mean(self._buffer['priority'])}")


### MuZero Buffer Implementation ###

class MuZeroGame:
    """ 儲存一場完整的 MuZero 遊戲軌跡 """
    def __init__(self):
        self.observations = []  # 環境觀察
        self.actions = []       # 執行的動作
        self.rewards = []       # 獲得的獎勵
        self.policies = []      # MCTS 產生的策略 (訪問次數分佈)
        self.dones = []         # 是否結束
        self.values = []        # 網路預測的價值
    
    def add_step(self, obs, action, reward, policy, done, value=None):
        """
        新增一步遊戲數據
        
        Args:
            obs: 環境觀察
            action: 執行的動作
            reward: 獲得的獎勵
            policy: MCTS 產生的策略
            done: 是否結束
            value: 網路預測的價值（可選）
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.dones.append(done)
        if value is not None:
            self.values.append(value)
    
    def __len__(self):
        return len(self.observations)


class MuZeroReplayBuffer:
    """ MuZero 專用的回放緩衝區 """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, game: MuZeroGame):
        """
        將一場完整的遊戲加入緩衝區
        
        Args:
            game: MuZeroGame 實例
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = game
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        從緩衝區中隨機抽樣
        
        Args:
            batch_size: 抽樣大小
            
        Returns:
            遊戲軌跡列表
        """
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
