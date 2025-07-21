import math
import random
import torch
import numpy as np
from copy import deepcopy


class MCTSNode:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, state, parent=None, action=None, is_terminal=False):
        self.state = state  # The state of the environment (a 'Structure' object)
        self.parent = parent
        self.action = action
        self.children = []
        
        self.visit_count = 0
        self.total_reward = 0.0
        
        self.is_terminal = is_terminal
        self.untried_actions = None

    def set_untried_actions(self, actions):
        self.untried_actions = actions

    @property
    def q_value(self):
        """Mean reward of the node."""
        if self.visit_count == 0:
            return 0
        return self.total_reward / self.visit_count

    def uct_value(self, c_puct=1.0):
        """Calculates the UCT value for the node."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCT formula
        exploitation_term = self.q_value
        exploration_term = c_puct * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation_term + exploration_term

    def select_child(self, c_puct):
        """Selects the best child node based on UCT."""
        return max(self.children, key=lambda child: child.uct_value(c_puct))


class MCTSAgent:
    """MCTS Agent for vanilla and hybrid (DQN-assisted) search."""
    def __init__(self, env, algorithm, n_simulations, c_puct, dqn_agent=None, rollout_depth=5, gamma=0.99):
        self.env = env
        self.algorithm = algorithm
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dqn_agent = dqn_agent
        self.rollout_depth = rollout_depth
        self.gamma = gamma

        if self.algorithm == "HybridMCTS" and self.dqn_agent is None:
            raise ValueError("HybridMCTS requires a dqn_agent to be provided.")

    def search(self, root_state):
        """
        Performs MCTS search for a given state and returns the best action.
        """
        root = MCTSNode(root_state)

        # A node is terminal if the env says so, or if there are no legal actions
        is_terminal_state = self.env.is_terminal(root.state)
        legal_actions = [] if is_terminal_state else self.env.get_legal_actions(root.state)
        print('='*100)
        print(f"legal_actions: {legal_actions}")
        print('='*100)
        root.is_terminal = not bool(legal_actions) or is_terminal_state
        if root.is_terminal:
            return None # No actions to take from the start

        for _ in range(self.n_simulations):
            node = root
            
            # --- Corrected Selection & Expansion Phase ---
            while not node.is_terminal:
                if node.untried_actions is None:
                    # First time visiting this node, get its possible actions
                    node.set_untried_actions(self.env.get_legal_actions(node.state))

                # If the node has untried actions, it's not fully expanded
                if node.untried_actions:
                    # Expand the node by creating one new child
                    node = self._expand(node)
                    # This new child is the leaf for this simulation, so we break
                    break
                
                # If no untried actions, the node is fully expanded.
                # Select the best child to continue traversal, unless it's a leaf.
                if not node.children:
                    # Reached a leaf node that has no further moves
                    break
                node = node.select_child(self.c_puct)
            # --- End of Corrected Phase ---
            
            # 3. Simulation & 4. Backpropagation
            reward = self._simulate_and_evaluate(node)
            self._backpropagate(node, reward)
            
        # DEBUGGING LOGS =======================================================
        if root.children:
            self.env.logger.info("--- MCTS Root Children Stats ---")
            sorted_children = sorted(root.children, key=lambda c: c.action)
            for child in sorted_children:
                self.env.logger.info(
                    f"Action: {child.action}, "
                    f"Q-Value: {child.q_value:.4f}, "
                    f"Visit Count: {child.visit_count}, "
                    f"UCT Value: {child.uct_value(c_puct=self.c_puct):.4f}" # Use c_puct=0 to see pure Q-value
                )
            self.env.logger.info("------------------------------------")
        # ======================================================================

        if not root.children:
            return None 
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action

    def _expand(self, node):
        """
        Expands the tree by creating a new child node from an untried action.
        """
        action = node.untried_actions.pop()
        
        next_structure_state = deepcopy(node.state)
        _, _, done, _, _ = self.env.step(next_structure_state, action)
        
        child_node = MCTSNode(next_structure_state, parent=node, action=action, is_terminal=done)
        node.children.append(child_node)
        return child_node

    def _simulate_and_evaluate(self, node):
        """
        Performs a rollout from the given node to estimate its value.
        Switches between Vanilla MCTS and Hybrid MCTS logic.
        """
        if self.algorithm == "HybridMCTS":
            return self._hybrid_simulation(node.state)
        else: # Vanilla MCTS
            return self._vanilla_simulation(node.state)

    def _vanilla_simulation(self, state):
        """
        Performs a random rollout from the state until a terminal state is reached.
        """
        current_state = deepcopy(state)
        
        while True:
            # Check if current state is terminal
            if self.env.is_terminal(current_state):
                 _, final_reward, _, _, _ = self.env.step(current_state, action=None) # Get terminal reward
                 return final_reward

            legal_actions = self.env.get_legal_actions(current_state)
            if not legal_actions: # No more moves, terminal state
                _, final_reward, _, _, _ = self.env.step(current_state, action=None) # Get terminal reward
                return final_reward
            
            action = random.choice(legal_actions)
            _, reward, done, _, _ = self.env.step(current_state, action)
            
            if done:
                return reward


    def _hybrid_simulation(self, state):
        """
        Performs a short random rollout and then uses DQN to evaluate the final state.
        """
        current_state = deepcopy(state)
        total_rollout_reward = 0.0
        actions_taken = []
        for i in range(self.rollout_depth):
            if self.env.is_terminal(current_state):
                _, final_reward, _, _, _ = self.env.step(current_state, action=None)
                return final_reward

            legal_actions = self.env.get_legal_actions(current_state)
            if not legal_actions:
                _, final_reward, _, _, _ = self.env.step(current_state, action=None)
                return final_reward
            
            action = random.choice(legal_actions)
            _, reward, done, _, _ = self.env.step(current_state, action)
            actions_taken.append(action)
            total_rollout_reward += (self.gamma ** i) * reward

            if done:
                return total_rollout_reward
        
        # After rollout, estimate value with DQN
        graph_for_dqn = deepcopy(current_state)

        # To get the graph features required by the GNN, we need to run a static analysis.
        from Structure import check
        try:
            # Run the full analysis process to get 'static_response_features'
            load_cases, responses = check.get_response(graph_for_dqn, self.env.code_analysis_dir)
            _, static_features, _ = check.process_response(graph_for_dqn, load_cases, responses)
        except Exception as e:
            # If the analysis itself fails, it's a very bad state.
            self.env.logger.error(f"Analysis failed during hybrid simulation: {e}")
            return -100.0  # Return a very low value for designs that cause errors.

        # Now, initialize the graph with the real features. Dynamic features are not needed for this evaluation.
        graph_for_dqn.init_graph_GraphRL(static_features, None)
        
        # Move the graph object's tensors to the same device as the DQN agent.
        graph_to_evaluate = graph_for_dqn.graph.to(self.dqn_agent.device)

        state_value_estimate = self.dqn_agent.get_state_value(graph_to_evaluate)
        final_value = total_rollout_reward + (self.gamma ** self.rollout_depth) * state_value_estimate
        self.env.logger.info('='*100)
        self.env.logger.info(f"Hybrid Sim: Rollout Reward={total_rollout_reward:.4f}, DQN Value={state_value_estimate:.4f}, Final Value={final_value:.4f}")
        self.env.logger.info(f"Hybrid Sim: Actions Taken={actions_taken}")
        self.env.logger.info('='*100)

        return total_rollout_reward + (self.gamma ** self.rollout_depth) * state_value_estimate


    def _backpropagate(self, node, reward):
        """
        Updates the visit counts and total rewards of nodes up the tree.
        """
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent      


### MuZero MCTS Implementation ###

class MuZeroNode:
    """ MuZero MCTS 的節點類別 """
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def muzero_ucb_score(parent: MuZeroNode, child: MuZeroNode, 
                     c1: float = 1.25, c2: float = 19652) -> float:
    """ 
    MuZero 的 UCB 分數計算 
    基於論文中的 PUCT (Polynomial Upper Confidence Trees) 公式
    """
    pb_c = math.log((parent.visit_count + c2 + 1) / c2) + c1
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def run_muzero_mcts(root_hidden_state: torch.Tensor, network, 
                    num_simulations: int, num_actions: int, discount: float = 0.99):
    """
    執行 MuZero MCTS 搜尋
    支援動態動作空間
    
    Args:
        root_hidden_state: 根節點的隱藏狀態
        network: MuZero 神經網路
        num_simulations: 模擬次數
        num_actions: 當前環境的動作數量
        discount: 折扣因子
    
    Returns:
        visit_counts: 根節點各子節點的訪問次數
    """
    # 1. 建立根節點
    root = MuZeroNode(0)
    
    # 使用 network 的 predict 方法獲取根節點的策略和價值
    with torch.no_grad():
        policy_logits, value = network.predict(root_hidden_state, num_actions)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)
    
    # 2. 擴展根節點
    for action in range(num_actions):
        root.children[action] = MuZeroNode(prior=policy_probs[action].item())
    root.hidden_state = root_hidden_state
    
    # 3. 進行模擬
    for simulation in range(num_simulations):
        node = root
        search_path = [node]
        
        # a. Selection - 沿著樹往下走，選擇 UCB 分數最高的節點
        while node.expanded():
            action, node = max(node.children.items(), 
                             key=lambda item: muzero_ucb_score(node, item[1]))
            search_path.append(node)
        
        # 獲取選擇的動作
        parent = search_path[-2]
        action_taken = None
        for action, child in parent.children.items():
            if child is node:
                action_taken = action
                break
        
        # b. Expansion & Simulation - 使用神經網路進行「想像」
        with torch.no_grad():
            # 用 dynamics 網路得到下一步的 reward 和 hidden_state
            action_tensor = torch.tensor([[action_taken]], dtype=torch.long)
            reward, next_hidden_state = network.dynamics(parent.hidden_state, action_tensor, num_actions)
            
            # 用 prediction 網路得到下一步的 policy 和 value
            policy_logits, value = network.predict(next_hidden_state, num_actions)
            policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)
        
        # 擴展新節點
        for action in range(num_actions):
            node.children[action] = MuZeroNode(prior=policy_probs[action].item())
        node.hidden_state = next_hidden_state
        node.reward = reward.item()
        
        # c. Backpropagation - 更新路徑上所有節點的統計值
        current_value = value.item()
        for node_in_path in reversed(search_path):
            node_in_path.value_sum += current_value
            node_in_path.visit_count += 1
            # 應用折扣因子和節點獎勵
            current_value = node_in_path.reward + discount * current_value
    
    # 回傳根節點的訪問次數分佈，作為訓練策略的目標
    # 確保返回的 visit_counts 長度等於 num_actions
    visit_counts = torch.zeros(num_actions, dtype=torch.float32)
    for action, child in root.children.items():
        if action < num_actions:  # 確保動作索引在有效範圍內
            visit_counts[action] = child.visit_count
    
    return visit_counts      