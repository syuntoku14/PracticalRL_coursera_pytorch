def evaluateMP(p, agent, env):
    """Plays an a game from start till done, returns per-game rewards """

    state = env.reset()

    total_reward = 0
    while True:
        state = torch.tensor(state).unsqueeze(0)
        action = agent.sample_actions(agent(state))[0]
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done: break

    total_reward
    return total_reward
