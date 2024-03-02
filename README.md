## What is This Repository?

This is the code repository for my honors thesis!

My thesis title is: Using Deep Learning Techniques to Find the 4D Slice Genus of a Knot, with abstract:

Deep reinforcement learning (DRL) has proven to be exceptionally effective in addressing challenges 
related to pattern recognition and problem-solving, particularly in domains where human intuition faces limitations. 
Within the field of knot theory, a significant obstacle lies in the construction of minimal-genus slice surfaces for knots of varying complexity. 
This thesis presents a new approach harnessing the capabilities of DRL to address this challenging problem. 
By employing braid representations of knots, our methodology involves training reinforcement learning agents to generate minimal-genus slice surfaces. 
The agents achieve this by identifying optimal sequences of braid transformations with respect to a defined objective function.

Ultimately, we used PPO to try and find the minimal slice genus of a knot. You can find all my code here, including my implementation of PPO! 

(If you are interested in reading my thesis, you can find it [here](https://scholarsarchive.byu.edu/cgi/preview.cgi?article=1352&context=studentpub_uht)!)

## Breakdown of File Structure

Here is a break down of the file structure in this repository. Hopefully it is helpful!

### `gym` and `gym-knot`

For this project, we utilized a custom `gym` environment that allowed our agents to easily interact with our desired problem. In order to use a custom gym environment, you need to 'register' it in the proper way. These files are how I registered it. There are two of them, though I'm not terribly sure which one is the real one (because I don't think you need two). I did this some time ago and cannot remember which made it work. Anywho, if you are interested in how to 'register' your own custom environments, it the [documentation](https://www.gymlibrary.dev/content/environment_creation/) is pretty helpful.

### `result_csv`

This folder contains lots of folders that contain `.csv` files of the results we achieved during this project. Each folder represents a different experiment we ran. Most of the folders are named something to the effect of `sol_n_m`, where `n` is the lower bound for the crossing number used in the experiment, and `m` is the upper bound. Most folders contain four `.csv` files, each corresponding to a different agent ran on a different gpu (we typically ran our experiment on four gpu's in parallel). The `.csv` file will contain the braid the agent was trying to solve, what epoch the braid was solved at, which step of the epoch the solution was found, the reward the agent achieved, and the sequence of moves the agent took to solve the braid. If there are less than four `.csv` files in a folder, that is because there are some agents that did not solve any of the braids it was presented with during the training process.

### `runs`

This folder contains the tensorboard files that we generated during our training. Inside this folder is a plethra of other folders that contain the results to specific experiments. There are a LOT of files here.

### `tensorboard_stable/updated_logs`

One thing we also tried during our experimentation was using a PPO algorithm created by Stable Baselines. We ended up not doing much with this, but during our experimentation with it, we had it generate tensorboard files so we could see how well it was doing (and thus compare it to our home-built model). These files are kept more as a legacy than anything else.

### `training_files`

This folder is probably not appropriately named because it contains the saved weights for our model. Most of these weights will be of little use. In fact, the only ones that will be truly helpful are the weights named `knot_gpu_optim_Large_[number].pt`. In this instance, `number` corresponds to the upper bound crossing number used (see `result_csv` section for more information). So, theoretically, if you wanted to use our results and try your hand at some knots of crossings 2-6, any file with `number` greater than 6 should solve the problem! You would just need to download our code and load in the appropriate `.pt` file.

### Assorted Python Files

The rest of the files found in this repository are our various python files we used to train our agent. The most important file is `knot_gpu_Large.py`. This file contains the model architecture that we used to get the results found in my thesis (see link in What is This Respository?). Inside of this file is all of the code for our model and training loops. All other files were previous iterations of our experiments.

Another file of interest is `knot_jobscript`. This file is a bash script that we could activate from the terminal to get the model training on BYU's SuperComputer. It is not a necessary file, but can give some insight into setting up the training process.
