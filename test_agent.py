import torch
import time

import snake_game
import agent
import cv2


if __name__ == "__main__":
    game = snake_game.SnakeGameAI()
    agent = agent.Agent()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = model.Linear_QNet(11, 256, 4)
    agent.model.load_state_dict(torch.load("./model/model_1.pth"))
    agent.model.eval()
    # model.to(device)  # Move our model to the gpu memory

    game_over = False
    while not game_over:

        # get state
        state = agent.get_state(game)

        # get move
        final_move = [0, 0, 0, 0]  # [Right, Up, Left, Down]
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        prediction = agent.model(state_tensor)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        # perform move and get new state
        reward, game_over, score = game.play_step(action=agent.move_list2str(final_move), visuals=True)
        print(reward, game_over, score)
        cv2.waitKey(20)
