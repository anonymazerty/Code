import os
import json
import numpy as np
from uuid import UUID
import pandas as pd
import torch
from app.agents.dqn_agent import DQNAgent
from app.agents.a2c_agent import A2CAgent
from app.agents.a3c_agent import A3CAgent
from app.agents.sarsa_agent import SARSAAgent
from app.environments.custom_env import CustomEnv
from app.utils.utilities import get_best_state_inference, compute_reward
from app.schemas.composition import QuizCompositionRequest


def compose_quiz(req: QuizCompositionRequest, uuid: UUID) -> str:
    """
    Compose a quiz based on Teacher goals.

    Args:
        req (QuizCompositionRequest): Request object containing quiz composition parameters.
        uuid (UUID): Unique identifier for the request.
    Returns:
        list: path to the generated quiz data file.
    """
    # create a directory for the data if it doesn't exist
    if not os.path.exists('output_quizzes'):
        os.makedirs('output_quizzes')

    best_quiz_id, targetMatch = inference(req, uuid)

    quizzes_df = pd.read_csv(f"data/{req.dataUUID}/quizzes_{req.dataUUID}.csv")
    best_quiz_row = quizzes_df[quizzes_df['quiz_id'] == best_quiz_id]

    best_quiz_dict = best_quiz_row.to_dict(orient='records')[0]
    best_quiz_dict["quiz_id"] = int(best_quiz_id)

    best_quiz_dict['uuid'] = str(uuid)
    best_quiz_dict['dataUUID'] = str(req.dataUUID)
    best_quiz_dict['targetMatch'] = str(targetMatch)

    os.makedirs(f'output_quizzes/{uuid}')
    with open(f'output_quizzes/{uuid}/request_{uuid}.yml', 'w') as f:
        for key, value in req.dict().items():
            f.write(f"{key}: {value}\n")

    output_path = f'output_quizzes/{uuid}/best_quiz_{uuid}.json'
    with open(output_path, 'w') as f:
        json.dump(best_quiz_dict, f, indent=4)

    # IMPORTANT: return dict + path (so UI doesn't depend on filesystem paths)
    return {"best_quiz": best_quiz_dict, "PathToQuiz": output_path}


class Args:
    def __init__(self, uuid: UUID, pathToModel: str, alfa: float = 0.5):
        self.alfa = alfa
        self.uuid = uuid
        self.pathToModel = pathToModel

        self.set_agent_type()
        self.reward_threshold = 0.85
        self.max_iterations = 100

        self.max_episodes = 5000
        self.gamma = 0.95
        self.lr = 0.0005
        self.batch_size = 128
        self.eps = 1.0
        self.eps_decay = 0.997
        self.eps_min = 0.05
        self.target_sync_freq = 1000

        self.set_num_topics()
        self.set_universe_size()
        self.quiz_size = 10
        self.generator_type = 'uniform'

        self.seed = 23

    def set_num_topics(self):
        # read the number of topics from the yaml file related to the UUID
        try:
            with open(f'data/{self.uuid}/request_{self.uuid}.yml', 'r') as f:
                for line in f:
                    if 'listTopics' in line:
                        topics = line.split(':')[1].strip()
                        if topics:
                            cleaned = topics.strip().strip('[]').strip()
                            if not cleaned:
                                self.num_topics = 0
                            else:
                                self.num_topics = len([t for t in cleaned.split(',') if t.strip()])
                                break
                    elif 'numTopics' in line:
                        self.num_topics = int(line.split(':')[1].strip())
                        break
        except FileNotFoundError:
            raise ValueError(f"Warning: Request file for UUID {self.uuid} not found.")

    def set_agent_type(self):
        if 'dqn' in self.pathToModel:
            agent_type = 'dqn'
        elif 'a2c' in self.pathToModel:
            agent_type = 'a2c'
        elif 'a3c' in self.pathToModel:
            agent_type = 'a3c'
        elif 'sarsa' in self.pathToModel:
            agent_type = 'sarsa'
        else:
            raise ValueError(f"Unknown agent type in path: {self.pathToModel}")
        self.agent_type = agent_type

    def set_universe_size(self):
        try:
            with open(f'data/{self.uuid}/request_{self.uuid}.yml', 'r') as f:
                for line in f:
                    if 'numQuizzes' in line:
                        self.universe_size = int(line.split(':')[1].strip())
                        break
        except FileNotFoundError:
            raise ValueError(f"Warning: Request file for UUID {self.uuid} not found.")


def run_agent(env: CustomEnv, agent, start_state: int) -> list:
    steps = 0
    done = False
    env.state = start_state
    all_states = [env.state]
    while not done:
        steps += 1
        action, _, _ = agent.get_action(env.universe[env.state], epsilon=0.0)
        next_state, _, done, _, _, _ = env.step(action, steps)
        env.state = next_state
        all_states.append(env.state)
    print(f"Total steps taken in the run: {steps}")
    return all_states


def inference(req: QuizCompositionRequest, uuid: UUID) -> tuple[int, float]:
    args = Args(req.dataUUID, req.pathToModel, req.alfaValue)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"data/{args.uuid}/universe_{args.uuid}.json", "r") as f:
        universe = json.load(f)
        universe = np.array(universe, dtype=np.float32)

    targets = [req.teacherTopic, req.teacherLevel]
    targets = [np.array(targets[0]), np.array(targets[1])]

    print(f"Universe shape: {universe.shape}")
    print(f"Targets shape: {targets[0].shape}, {targets[1].shape}")
    print(f"Starting inference for {uuid}...")

    start_state = None
    if getattr(req, "startQuizId", None) is not None:
        start_state = int(req.startQuizId)
        if start_state < 0 or start_state >= args.universe_size:
            raise ValueError(
                f"startQuizId out of range: {start_state} (universe_size={args.universe_size})"
            )
    else:
        start_state = int(np.random.choice(args.universe_size, 1)[0])

    env = CustomEnv(
        universe=universe,
        target_dim1=targets[0],
        target_dim2=targets[1],
        max_iterations=args.max_iterations,
        num_topics=args.num_topics,
        alfa=args.alfa,
        reward_threshold=args.reward_threshold,
        state=start_state
    )

    if args.agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            device=device,
            lr=args.lr,
            gamma=args.gamma,
            target_sync_freq=args.target_sync_freq,
            batch_size=args.batch_size
        )
    elif args.agent_type == 'a2c':
        agent = A2CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                        lr=args.lr, gamma=args.gamma)
    elif args.agent_type == 'a3c':
        agent = A3CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                        lr=args.lr, gamma=args.gamma)
    elif args.agent_type == 'sarsa':
        agent = SARSAAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                           lr=args.lr, gamma=args.gamma, eps=args.eps, eps_decay=args.eps_decay,
                           eps_min=args.eps_min)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    agent.load(f"{args.pathToModel}/agent_alfa_{args.alfa}_bias.pth")

    if args.agent_type == 'a3c':
        agent.actor_critic.eval()
    elif args.agent_type == 'a2c':
        agent.global_actor_critic.eval()
    else:
        agent.model.eval()

    inference_states = run_agent(env, agent, start_state)
    picked_quiz_id = inference_states[get_best_state_inference(universe[inference_states], targets, args.alfa)]

    print(f"Inference completed!")
    print(f"Best quiz ID selected: {picked_quiz_id}")

    targetMatch = compute_reward(args.alfa, universe[picked_quiz_id], targets)
    print(f"TargetMatch value of: {targetMatch}")
    return picked_quiz_id, targetMatch
