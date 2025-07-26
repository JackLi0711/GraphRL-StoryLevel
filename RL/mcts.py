import math
import random
import torch
import numpy as np
from copy import deepcopy
from typing import Sequence


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
    def __init__(self, prior: float, hidden_state=None, structure=None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.hidden_state = hidden_state   # 從 network.represent 或 dynamics 回傳
        self.reward = 0.0
        self.structure = structure         # 真實 env state

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


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
    return prior_score + value_score, prior_score, value_score


def run_muzero_mcts(
    root_hidden_state: torch.Tensor,
    network,
    env,
    root_structure,
    num_simulations: int,
    discount: float = 0.99
) -> torch.Tensor:

    logger = getattr(env, 'logger', None)
    if logger is None:
        import logging
        logger = logging.getLogger("muzero_mcts_debug")
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    root = MuZeroNode(0.0, hidden_state=root_hidden_state, structure=root_structure)

    # ---------- expand root ----------
    legal0 = env.get_legal_actions(root.structure)
    with torch.no_grad():
        logits, _ = network.predict(root.hidden_state, None)
        probs = torch.softmax(logits.squeeze(), dim=-1)

    for a in legal0:
        root.children[a] = MuZeroNode(prior=probs[a].item())

    # -------- Dirichlet exploration noise (MuZero) --------
    if len(legal0) > 0:
        epsilon = 0.25
        alpha   = 0.3
        dirichlet_noise = np.random.dirichlet([alpha] * len(legal0)).astype(float)
        for i, a in enumerate(legal0):
            root.children[a].prior = (1 - epsilon) * root.children[a].prior + epsilon * dirichlet_noise[i]
    logger.debug(f"[ROOT EXPAND] legal0: {legal0}")
    logger.debug(f"[ROOT EXPAND] root.children.keys(): {list(root.children.keys())}")

    # ---------- simulations ----------
    for sim in range(num_simulations):
        node  = root
        path  = [node]
        actions_taken = []

        # a) selection
        while node.expanded():
            for a in node.children.keys():
                logger.debug(f"node.children[{a}].prior: {node.children[a].prior}")
                logger.debug(f"node.children[{a}].visit_count: {node.children[a].visit_count}")
                logger.debug(f"node.children[{a}].value(): {node.children[a].value()}")
                uct_score, prior_score, value_score = muzero_ucb_score(path[-1], node.children[a])
                logger.debug(f"node.children[{a}].uct_score: {uct_score}, prior_score: {prior_score}, value_score: {value_score}")
            action, node = max(node.children.items(),
                               key=lambda kv: muzero_ucb_score(path[-1], kv[1])[0])
            path.append(node)
            actions_taken.append(action)
            logger.debug(f"[SIM {sim}] Selection {len(actions_taken)}:  path actions: {actions_taken}")
            logger.debug(f"[SIM {sim}] Selection {len(actions_taken)}:  path node.children.keys(): {list(node.children.keys())}")
        logger.debug(f"[SIM {sim}] Selection {len(actions_taken)}:  path actions: {actions_taken}")

        # 若 root 沒合法動作直接 break
        if len(path) == 1:
            logger.debug(f"[SIM {sim}] No legal actions at root, break.")
            break

        parent = path[-2]
        action = actions_taken[-1]
        logger.debug(f"[SIM {sim}] Parent.children.keys(): {list(parent.children.keys())}")
        logger.debug(f"[SIM {sim}] Parent legal_actions: {env.get_legal_actions(parent.structure)}")
        logger.debug(f"[SIM {sim}] Action taken: {action}")

        # b) env.step & model rollout
        before = deepcopy(parent.structure)
        new_structure, _, done, _, _ = env.step(deepcopy(parent.structure), action)
        logger.debug(f"[SIM {sim}] Expansion done: {done}")
        # assert parent.structure == before, "env.step() mutated the input!"

        with torch.no_grad():
            a_tensor = torch.tensor([[action]], dtype=torch.long, device=root_hidden_state.device)
            reward_pred, next_hidden = network.dynamics(parent.hidden_state, a_tensor, None)
            logits, value = network.predict(next_hidden, None)
            probs = torch.softmax(logits.squeeze(), dim=-1)

        # 更新目前 node
        node.hidden_state = next_hidden
        node.reward       = reward_pred.item()
        node.structure    = deepcopy(new_structure)

        # expand children with *its* legal actions, only if not done
        node.children.clear()
        legal = env.get_legal_actions(new_structure)
        is_terminal = done or (not legal)
        if is_terminal:
            node.children.clear()
            logger.debug(f"[SIM {sim}] Node is terminal (done or no legal actions), no children expanded.")
        else:
            for a in legal:
                node.children[a] = MuZeroNode(prior=probs[a].item())
            logger.debug(f"[SIM {sim}] Node expand legal: {legal}")
            logger.debug(f"[SIM {sim}] Node.children.keys(): {list(node.children.keys())}")

        # c) back-prop
        bootstrap = value.item()
        for n in reversed(path):
            n.value_sum  += bootstrap
            n.visit_count += 1
            bootstrap = n.reward + discount * bootstrap

    # ---------- collect visit counts ----------
    visit_counts = torch.zeros(len(legal0), device=root_hidden_state.device)
    for idx, a in enumerate(legal0):
        print(f'root.children[{a}]: {root.children[a].visit_count}')
        visit_counts[idx] = root.children[a].visit_count if a in root.children else 0
    logger.debug(f"[FINAL] legal0: {legal0}")
    logger.debug(f"[FINAL] root.children.keys(): {list(root.children.keys())}")
    logger.debug(f"[FINAL] visit_counts: {visit_counts.tolist()}")
    
    return visit_counts
