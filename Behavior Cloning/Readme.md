# CNNs for Behavioral Cloning in Pong
This project trains a convolutional neural network (CNN) to play the Atari Pong game using behavioral cloning. The goal is to replicate expert actions by mapping game state observations (images) to corresponding actions. The trained model controls an AI agent that competes against the computer, aiming to score 21 points first.

In this game, each player can execute one of 6 possible actions at each timestep (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, and LEFTFIRE). The goal is to execute the best action at each timestep based on the current state of the game.
Using PyTorch, we implement and train a simple CNN to map the images to their corresponding expert actions. We implement a control policy that dictates how the agent behaves in differnet situations, and the approach of
training this NN in a supervised manner from expert trajectories is called behavioral cloning. 

To get started, first download the following files:

[actions](https://s3.amazonaws.com/jrwprojects/pong_actions.pt) <br>
[observations](https://s3.amazonaws.com/jrwprojects/pong_observations.pt)
