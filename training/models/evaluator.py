import json  # Import for saving results
import os
import time

import numpy as np
import torch


class Evaluator:  # [ElegantRL.2022.01.01]
    """
    Evaluator class for monitoring and evaluating the agent's performance.
    This class handles recording metrics, saving models, and plotting learning curves.
    It includes enhanced error handling for better debugging.
    """

    def __init__(self, cwd, agent_id, eval_env, args):
        """
        Initialize the Evaluator.

        Args:
        cwd (str): Current working directory for saving files.
        agent_id (int): ID of the agent being evaluated.
        eval_env: The environment for evaluation.
        args: Configuration arguments.
        """
        try:
            self.recorder = (
                list()
            )  # List to store metrics: total_step, r_avg, r_std, obj_c, ...
            self.recorder_path = f"{cwd}/recorder.npy"  # Path to save recorder data

            self.cwd = cwd
            self.agent_id = agent_id
            self.eval_env = eval_env
            self.eval_gap = args.eval_gap  # Time gap between evaluations
            self.eval_times1 = max(
                1, int(args.eval_times / np.e)
            )  # First evaluation times
            self.eval_times2 = max(
                0, int(args.eval_times - self.eval_times1)
            )  # Second evaluation times
            self.target_return = args.target_return  # Target return for the agent

            self.r_max = -np.inf  # Maximum reward achieved so far
            self.eval_time = 0  # Last evaluation time
            self.used_time = 0  # Total used time
            self.total_step = 0  # Total steps
            self.start_time = time.time()  # Start time of evaluation
            print(
                f"{'#' * 80}\n"
                f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
                f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                f"{'expR':>8}{'objC':>7}{'etc.':>7}"
            )
        except Exception as e:
            print(f"Error in __init__: {e}")  # Debug message for initialization errors

    def evaluate_save_and_plot(
        self, act, steps, r_exp, log_tuple
    ) -> (bool, bool):  # 2021-09-09
        """
        Evaluate the agent, save if improved, and plot results.

        Args:
        act: The actor model.
        steps (int): Number of steps taken.
        r_exp (float): Expected reward.
        log_tuple: Additional logging metrics.

        Returns:
        if_reach_goal (bool): Whether the target return is reached.
        if_save (bool): Whether to save the model.
        """
        try:
            self.total_step += steps  # Update total training steps

            if time.time() - self.eval_time < self.eval_gap:
                if_reach_goal = False
                if_save = False
            else:
                self.eval_time = time.time()

                # Evaluate first time
                rewards_steps_list = [
                    get_episode_return_and_step(self.eval_env, act)
                    for _ in range(self.eval_times1)
                ]
                rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)

                r_avg, s_avg = rewards_steps_ary.mean(
                    axis=0
                )  # Average of episode return and step

                # Evaluate second time if improved
                if r_avg > self.r_max:
                    rewards_steps_list.extend(
                        [
                            get_episode_return_and_step(self.eval_env, act)
                            for _ in range(self.eval_times2)
                        ]
                    )
                    rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
                    r_avg, s_avg = rewards_steps_ary.mean(axis=0)

                r_std, s_std = rewards_steps_ary.std(axis=0)  # Standard deviation

                # Save the policy network if improved
                if_save = r_avg > self.r_max
                if if_save:
                    self.r_max = r_avg
                    act_path = (
                        f"{self.cwd}/actor_{self.total_step:08}_{self.r_max:09.3f}.pth"
                    )
                    torch.save(act.state_dict(), act_path)
                    print(
                        f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                    )

                self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))

                if_reach_goal = bool(self.r_max > self.target_return)
                if if_reach_goal and self.used_time is None:
                    self.used_time = int(time.time() - self.start_time)
                    print(f"Goal reached in {self.used_time} seconds.")

                print(
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} | {r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} | {r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}"
                )

                if len(self.recorder) > 0:
                    np.save(self.recorder_path, self.recorder)
                    save_learning_curve(self.recorder, self.cwd, "Learning Curve")
            return if_reach_goal, if_save
        except Exception as e:
            print(f"Error in evaluate_save_and_plot: {e}")
            return False, False  # Debug message for errors

    def save_results(self, metrics, filename):
        """Save metrics to a JSON file."""
        try:
            file_path = os.path.join(self.cwd, filename)
            with open(file_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {file_path}")
        except Exception as e:
            print(f"Error in save_results: {e}")

    def evaluate(self, model, X, y):
        # Placeholder for evaluation logic; implement as needed
        try:
            return model.evaluate(X, y)
        except Exception as e:
            print(f"Error in evaluate: {e}")
            return None

    def save_or_load_recoder(self, if_save):
        try:
            if if_save:
                np.save(self.recorder_path, self.recorder)
            elif os.path.exists(self.recorder_path):
                recorder = np.load(self.recorder_path)
                self.recorder = [tuple(i) for i in recorder]
                self.total_step = self.recorder[-1][0] if self.recorder else 0
        except Exception as e:
            print(f"Error in save_or_load_recoder: {e}")


"""util"""


def get_episode_return_and_step(env, act) -> (float, int):  # [ElegantRL.2022.01.01]
    """
    Get the return and step for an episode.

    Args:
    env: The environment.
    act: The actor model.

    Returns:
    episode_return (float): Total return of the episode.
    episode_step (int): Number of steps in the episode.
    """
    try:
        max_step = env.max_step
        if_discrete = env.if_discrete
        device = next(act.parameters()).device

        state = env.reset()
        episode_return = 0.0
        for episode_step in range(max_step):
            s_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            a_tensor = act(s_tensor)
            if if_discrete:
                a_tensor = a_tensor.argmax(dim=1)
            action = a_tensor.detach().cpu().numpy()[0]
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break
        return episode_return, episode_step + 1
    except Exception as e:
        print(f"Error in get_episode_return_and_step: {e}")
        return 0, 0  # Return default values on error for debugging


def save_learning_curve(
    recorder=None,
    cwd=".",
    save_title="learning curve",
    fig_name="plot_learning_curve.jpg",
):
    """
    Save the learning curve plot.

    Args:
    recorder: Array of recorded metrics.
    cwd: Current working directory.
    save_title: Title for the plot.
    fig_name: Name of the figure file.
    """
    try:
        if recorder is None:
            recorder = np.load(f"{cwd}/recorder.npy")

        recorder = np.array(recorder)
        steps = recorder[:, 0]
        r_avg = recorder[:, 1]
        r_std = recorder[:, 2]
        r_exp = recorder[:, 3]
        obj_c = recorder[:, 4]
        obj_a = recorder[:, 5]

        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2)
        ax00 = axs[0]
        ax00.cla()
        ax01 = axs[0].twinx()
        ax01.set_ylabel("Explore AvgReward", color="darkcyan")
        ax01.plot(steps, r_exp, color="darkcyan", alpha=0.5)
        ax00.set_ylabel("Episode Return", color="lightcoral")
        ax00.plot(steps, r_avg, label="Episode Return", color="lightcoral")
        ax00.fill_between(
            steps, r_avg - r_std, r_avg + r_std, facecolor="lightcoral", alpha=0.3
        )
        ax00.grid()

        ax10 = axs[1]
        ax10.cla()
        ax11 = axs[1].twinx()
        ax11.set_ylabel("objC", color="darkcyan")
        ax11.fill_between(steps, obj_c, facecolor="darkcyan", alpha=0.2)
        ax10.set_xlabel("Total Steps")
        ax10.set_ylabel("objA", color="royalblue")
        ax10.plot(steps, obj_a, label="objA", color="royalblue")
        for plot_i in range(6, recorder.shape[1]):
            other = recorder[:, plot_i]
            ax10.plot(steps, other, label=f"{plot_i}", color="grey", alpha=0.5)
        ax10.legend()
        ax10.grid()

        plt.title(save_title, y=2.3)
        plt.savefig(f"{cwd}/{fig_name}")
        plt.close("all")
    except Exception as e:
        print(f"Error in save_learning_curve: {e}")
