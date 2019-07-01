import argparse
import logging
from collections import namedtuple, defaultdict
import numpy as np
import datetime
import pickle
import os

from gym import spaces, Env

# EpisodeStats only used to conveniently track training performance
EpisodeStats = namedtuple("Stats", ["episode_rewards", "expected_rewards"])


class SigmoidMultiValAction(Env):
    """
    Sigmoid reward with multiple actions
    """

    def _sig(self, x):
        """ Simple sigmoid """
        return 1 / (1 + np.exp(-self.slope*(x-self.shift)))

    def __init__(self,
                 n_steps: int=10,
                 n_actions: int=3,
                 seed: bool=0,
                 noise: bool=False) -> None:

        super().__init__()
        self.n_steps = n_steps
        self.action_space = spaces.Discrete(n_actions)
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        self.slope = 1
        self.shift = n_steps / 2
        self.reward_range = (0, 1)
        self._c_step = 0
        self.noise = noise
        self.observation_space = spaces.Box(low=np.array([0, self.n_steps/2 - 5*self.n_steps/2,
                                                          -2, -1]),
                                            high=np.array([self.n_steps, self.n_steps/2 + 5*self.n_steps/2,
                                                           2, n_actions]))
        self.logger = logging.getLogger(self.__str__())
        self._prev_state = None

    def step(self, action: int):
        """
        Advance on step forward
        :param action: int
        :return: next state, reward, done, misc
        """
        r = 1-np.abs(self._sig(self._c_step) - (action / (self.action_space.n - 1)))
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget, self.shift, self.slope, action]
        prev_state = self._prev_state

        self.logger.debug("i: (s, a, r, s') / %d: (%s, %d, %5.2f, %2s)", self._c_step-1, str(prev_state),
                          action, r, str(next_state))
        self._c_step += 1
        self._prev_state = next_state
        return np.array(next_state), r, self._c_step > self.n_steps, {}

    def reset(self):
        """
        Reset the environment.
        Always needed before starting a new episode.
        :return: Start state
        """
        if self.noise:
            self.shift = self.rng.normal(self.n_steps/2, self.n_steps/4)
            self.slope = self.rng.choice([-1, 1]) * self.rng.uniform() * 2  # negative slope
        self._c_step = 0
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget, self.shift, self.slope, -1]
        self._prev_state = None
        self.logger.debug("i: (s, a, r, s') / %d: (%2d, %d, %5.2f, %2d)", -1, -1, -1, -1, -1)
        return np.array(next_state)


class QTable(dict):
    def __init__(self, n_actions, float_to_int=False, **kwargs):
        """
        Look up table for state-action values.

        :param n_actions: action space size
        :param float_to_int: flag to determine if state values need to be rounded to the closest integer
        """
        super().__init__(**kwargs)
        self.n_actions = n_actions
        self.float_to_int = float_to_int
        self.__table = defaultdict(lambda: np.zeros(n_actions))

    def __getitem__(self, item):
        try:
            table_state, table_action = item
            if self.float_to_int:
                table_state = map(int, table_state)
            return self.__table[tuple(table_state)][table_action]
        except ValueError:
            if self.float_to_int:
                item = map(int, item)
            return self.__table[tuple(item)]

    def __setitem__(self, key, value):
        try:
            table_state, table_action = key
            if self.float_to_int:
                table_state = map(int, table_state)
            self.__table[tuple(table_state)][table_action] = value
        except ValueError:
            if self.float_to_int:
                key = map(int, key)
            self.__table[tuple(key)] = value

    def __contains__(self, item):
        return tuple(item) in self.__table.keys()

    def keys(self):
        return self.__table.keys()


def make_epsilon_greedy_policy(Q: QTable, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.

    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.argwhere(  # random choice for tie-breaking only
            Q[observation] == np.amax(Q[observation])
        ).flatten())
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_episodes: int, type_: str):
    """
    Create epsilon decay schedule

    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_episodes: Total number of episodes to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_episodes)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_episodes - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_episodes - decay_start))])
    else:
        raise NotImplementedError


def update(Q: QTable, environment: SigmoidMultiValAction, policy: callable, alpha: float, discount_factor: float):
    """
    Q update
    :param Q: state-action value look-up table
    :param environment: environment to use
    :param policy: the current policy
    :param alpha: learning rate
    :param discount_factor: discounting factor
    """
    # Need to parse to string to easily handle list as state with defdict
    policy_state = environment.reset()
    episode_length, cummulative_reward = 0, 0
    expected_reward = np.max(Q[policy_state])
    while True:  # roll out episode
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        cummulative_reward += policy_reward
        episode_length += 1
        Q[[policy_state, policy_action]] = Q[[policy_state, policy_action]] + alpha * (
                (policy_reward + discount_factor * Q[[s_, np.argmax(Q[s_])]]) - Q[[policy_state, policy_action]])
        if policy_done:
            break
        policy_state = s_
    return Q, cummulative_reward, expected_reward, episode_length  # Q, cumulative reward


def greedy_eval_Q(Q: QTable, this_environment: SigmoidMultiValAction, nevaluations: int = 1):
    """
    Evaluate Q function greediely with epsilon=0

    :returns average cumulative reward, average expected Reward at start of episode
    """
    cumuls = []
    expects = []
    for _ in range(nevaluations):
        evaluation_state = this_environment.reset()
        cummulative_reward = 0
        expects.append(np.amax(Q[evaluation_state]))
        greedy = make_epsilon_greedy_policy(Q, 0, this_environment.action_space.n)
        while True:  # roll out episode
            evaluation_action = np.random.choice(list(range(this_environment.action_space.n)),
                                                 p=greedy(evaluation_state))
            s_, evaluation_reward, evaluation_done, _ = this_environment.step(evaluation_action)
            cummulative_reward += evaluation_reward
            if evaluation_done:
                break
            evaluation_state = s_
        cumuls.append(cummulative_reward)
    return np.mean(cumuls), np.mean(expects)  # cumulative reward


def q_learning(environment: SigmoidMultiValAction,
               num_episodes: int,
               discount_factor: float = 1.0,
               alpha: float = 0.5,
               epsilon: float = 0.1,
               float_state=True,
               epsilon_decay: str = 'const',
               decay_starts: int = 0,
               number_of_evaluations: int = 1):
    """
    Q-Learning algorithm
    :param environment: The environment to learn on
    :param num_episodes: Total number of episodes to train
    :param discount_factor: Discounting factor for future rewards
    :param alpha: Learning rate
    :param epsilon: Exploration fraction
    :param float_state: flag to decide if the Q-table has to handle float values in the states
    :param epsilon_decay: see get_decay_schedule
    :param decay_starts: see get_decay_schedule
    :param number_of_evaluations: Number of episodes used for evaluating the training performance
    :return: The learned Q-function as well as training statistics
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = QTable(env.action_space.n, float_state)  # TODO modify as necessary for your Q-table

    # Keeps track of episode lengths and rewards
    train_stats = EpisodeStats(
        episode_rewards=np.zeros(num_episodes),
        expected_rewards=np.zeros(num_episodes))
    train_reward = 0

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    for i_episode in range(num_episodes):
        epsilon = epsilon_schedule[i_episode]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {:>5d}/{}\tReward: {:>4.2f}".format(i_episode + 1, num_episodes, train_reward), end='')
        # TODO rollout episode following the current policy and update Q
        Q, _, _, _ = update(Q, environment, policy, alpha, discount_factor)

        # Keep track of training reward
        train_reward, train_expected_reward = greedy_eval_Q(Q, environment, nevaluations=number_of_evaluations)
        train_stats.episode_rewards[i_episode] = train_reward
        train_stats.expected_rewards[i_episode] = train_expected_reward

    print()  # needed as the prior prints all print to the same line
    return Q, train_stats


def zeroOne(stringput):
    """
    Helper to keep input arguments in [0, 1]
    """
    val = float(stringput)
    if val < 0 or val > 1.:
        raise argparse.ArgumentTypeError("%r is not in [0, 1]", stringput)
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tabular Q-learning example')
    parser.add_argument('-n', '--n-eps', dest='neps',
                        default=10000,
                        help='Number of episodes to roll out.',
                        type=int)
    parser.add_argument('--epsilon_decay',
                        choices=['linear', 'log', 'const'],
                        default='const',
                        help='How to decay epsilon from the given starting point to 0 or constant')
    parser.add_argument('--decay_starts',
                        type=int,
                        default=0,
                        help='How long to keep epsilon constant before decaying. '
                             'Only active if epsilon_decay log or linear')
    parser.add_argument('-r', '--repetitions',
                        default=1,
                        help='Number of repeated learning runs.',
                        type=int)
    parser.add_argument('-d', '--discount-factor', dest='df',
                        default=.99,
                        help='Discount factor',
                        type=zeroOne)

    # In a deterministic environment the learning rate can be simply set to 1 as we can completely trust all true
    # rewards we observe. So we don't need to worry if we accidentally update our q-value to far in any direction,
    # as the update direction will be exactly right.
    parser.add_argument('-l', '--learning-rate', dest='lr',
                        default=.125,
                        help='Learning rate',
                        type=float)

    # In such a limited state and action space, having a large exploration factor helps to quickly explore the whole
    # environment. Additionally due to the deterministic nature of the environment we quickly learn the true reward
    # landscape through high exploration values.
    parser.add_argument('-e', '--epsilon',
                        default=0.01,
                        help='Epsilon for the epsilon-greedy method to follow',
                        type=zeroOne)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Use debug output')
    parser.add_argument('-s', '--seed',
                        default=0,
                        type=int)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    q_func = None
    # tabular Q
    ds = datetime.date.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S_%f")
    random_agent = False
    folder = 'tabular_' + ds
    if not os.path.exists(folder):
        os.mkdir(folder)
        os.chdir(folder)
    for r in range(args.repetitions):
        logging.info('Round %d/%d', r + 1, args.repetitions)
        env = SigmoidMultiValAction()
        q_func, train_stats = q_learning(env,
                                         args.neps,
                                         discount_factor=args.df,
                                         alpha=args.lr,
                                         epsilon=args.epsilon,
                                         float_state=True,
                                         epsilon_decay=args.epsilon_decay,
                                         decay_starts=args.decay_starts,
                                         number_of_evaluations=1)
        fn = '%04d_%s-greedy-results-%s-%d_eps-%d_reps-seed_%d.pkl' % (r, str(args.epsilon), 'sig', args.neps,
                                                                       args.repetitions, args.seed)
        with open(fn.replace('results', 'q_func'), 'wb') as fh:  # TODO modify s.t. your Q function can be stored
            pickle.dump(dict(q_func), fh)
        with open(fn.replace('results', 'trueReward'), 'wb') as fh:
            pickle.dump(train_stats.episode_rewards, fh)
        with open(fn.replace('results', 'expectedReward'), 'wb') as fh:
            pickle.dump(train_stats.expected_rewards, fh)

        # Some post training Q-examples
        if r == args.repetitions - 1:
            logging.info('Example Q-function following greedy-policy:')
            for i in range(1):  # More examples only necessary for noisy environment
                print('#' * 120)
                current_state = env.reset()
                done = False
                while not done:
                    q_vals = q_func[current_state]
                    action = np.argmax(q_vals)  # type: int
                    print(current_state, q_vals, action)
                    current_state, _, done, _ = env.step(action)
