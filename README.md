Welcome to Multi-task model with learning rate schedule for XNLI, and GLUE.

Look into Multi_Task_training.ipynb for detailed comments on the functions and variables responsible for building a multi-task model with batch heterogenity, custom loss and accuracy functions.

Visit xtreme/ for multi task model code with learnable sampling policies using RL for experiments on XNLI, GLUE/ for the trainer code on GLUE data based on the Multi_task_training.iypnb implementation. mtmodel_policy_tasks.py contains the RL policy code to update weights for multiple tasks.
