from copy import deepcopy
import itertools
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym

import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger


def save_checkpoint(ac, ac_targ, pi_optimizer, q_optimizer, epoch, path, gradient_history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "actor": ac.pi.state_dict(),
        "q1": ac.q1.state_dict(),
        "q2": ac.q2.state_dict(),
        "target": ac_targ.state_dict(),
        "pi_opt": pi_optimizer.state_dict(),
        "q_opt": q_optimizer.state_dict(),
        "epoch": epoch,
        "gradient_history": gradient_history,
    }, path)


def load_checkpoint(ac, ac_targ, pi_optimizer, q_optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")

    ac.pi.load_state_dict(checkpoint["actor"])
    ac.q1.load_state_dict(checkpoint["q1"])
    ac.q2.load_state_dict(checkpoint["q2"])

    if "target" in checkpoint:
        ac_targ.load_state_dict(checkpoint["target"])
    else:
        ac_targ.load_state_dict(ac.state_dict())

    pi_optimizer.load_state_dict(checkpoint["pi_opt"])
    q_optimizer.load_state_dict(checkpoint["q_opt"])

    return checkpoint["epoch"], checkpoint.get("gradient_history", None)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )

        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.items()
        }


def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=1,
):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = list(itertools.chain(ac.q1.parameters(), ac.q2.parameters()))

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
    )

    var_counts = tuple(
        core.count_vars(module)
        for module in [ac.pi, ac.q1, ac.q2]
    )

    logger.log(
        "\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n"
        % var_counts
    )

    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    exp_name = logger_kwargs.get("exp_name", "sac_run")
    checkpoint_root = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_root, f"{exp_name}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    gradient_history = {}
    epoch_grad_buffer = defaultdict(list)

    def get_layer_grad_norms(module, prefix):
        grad_dict = {}

        for name, p in module.named_parameters():
            key = f"{prefix}.{name}"

            if p.grad is not None:
                grad_dict[key] = p.grad.data.norm(2).item()
            else:
                grad_dict[key] = np.nan

        return grad_dict

    def accumulate_gradients(grad_dict):
        for k, v in grad_dict.items():
            if not np.isnan(v):
                epoch_grad_buffer[k].append(v)

    def summarize_epoch_gradients():
        summary = {}

        for k, vals in epoch_grad_buffer.items():
            if len(vals) > 0:
                summary[k] = float(np.mean(vals))
            else:
                summary[k] = np.nan

        return summary

    def reset_epoch_gradient_buffer():
        epoch_grad_buffer.clear()

    def compute_loss_q(data):
        o = data["obs"]
        a = data["act"]
        r = data["rew"]
        o2 = data["obs2"]
        d = data["done"]

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(),
            Q2Vals=q2.detach().cpu().numpy(),
        )

        return loss_q, q_info

    def compute_loss_pi(data):
        o = data["obs"]

        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (alpha * logp_pi - q_pi).mean()

        pi_info = dict(
            LogPi=logp_pi.detach().cpu().numpy()
        )

        return loss_pi, pi_info

    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Found {checkpoint_path}")

        try:
            start_epoch, loaded_history = load_checkpoint(
                ac,
                ac_targ,
                pi_optimizer,
                q_optimizer,
                checkpoint_path,
            )

            if loaded_history is not None:
                gradient_history = loaded_history

            print(f"[CHECKPOINT] Loaded successfully from epoch {start_epoch}")

        except RuntimeError as e:
            print(f"[CHECKPOINT] Incompatible checkpoint, starting fresh.\nReason: {e}")
            start_epoch = 0
            gradient_history = {}

        except Exception as e:
            print(f"[CHECKPOINT] Could not load checkpoint, starting fresh.\nReason: {e}")
            start_epoch = 0
            gradient_history = {}

    logger.setup_pytorch_saver(ac)

    def update(data):
        q_optimizer.zero_grad()

        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()

        critic_gn = 0.0

        for p in q_params:
            if p.grad is not None:
                critic_gn += p.grad.data.norm(2).item() ** 2

        critic_gn = critic_gn ** 0.5

        q1_grads = get_layer_grad_norms(ac.q1, "critic.qf0")
        q2_grads = get_layer_grad_norms(ac.q2, "critic.qf1")

        accumulate_gradients(q1_grads)
        accumulate_gradients(q2_grads)

        q_optimizer.step()

        logger.store(
            CriticGradNorm=critic_gn,
            LossQ=loss_q.item(),
            **q_info,
        )

        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()

        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()

        actor_gn = 0.0

        for p in ac.pi.parameters():
            if p.grad is not None:
                actor_gn += p.grad.data.norm(2).item() ** 2

        actor_gn = actor_gn ** 0.5

        pi_grads = get_layer_grad_norms(ac.pi, "actor")
        accumulate_gradients(pi_grads)

        pi_optimizer.step()

        logger.store(
            ActorGradNorm=actor_gn,
            LossPi=loss_pi.item(),
            **pi_info,
        )

        for p in q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        """
        SAC-safe action function.

        This avoids:
        TypeError: act() takes 2 positional arguments but 3 were given

        Instead of calling ac.act(obs, deterministic), we call the policy
        directly, because SAC's policy supports deterministic=True/False.
        """

        obs_tensor = torch.as_tensor(o, dtype=torch.float32)

        with torch.no_grad():
            a, _ = ac.pi(
                obs_tensor,
                deterministic=deterministic,
                with_logprob=False,
            )

        return a.cpu().numpy()

    def test_agent():
        for _ in range(num_test_episodes):
            o, _ = test_env.reset()

            d = False
            ep_ret = 0
            ep_len = 0

            while not (d or ep_len == max_ep_len):
                a = get_action(o, deterministic=True)
                o, r, terminated, truncated, _ = test_env.step(a)

                d = terminated or truncated
                ep_ret += r
                ep_len += 1

            logger.store(
                TestEpRet=ep_ret,
                TestEpLen=ep_len,
            )

    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    o, _ = env.reset()
    ep_ret = 0
    ep_len = 0

    reset_epoch_gradient_buffer()

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, deterministic=False)
        else:
            a = env.action_space.sample()

        o2, r, terminated, truncated, _ = env.step(a)

        d = terminated or truncated

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or ep_len == max_ep_len:
            logger.store(
                EpRet=ep_ret,
                EpLen=ep_len,
            )

            o, _ = env.reset()
            ep_ret = 0
            ep_len = 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            gradient_history[epoch] = summarize_epoch_gradients()
            reset_epoch_gradient_buffer()

            if epoch % 10 == 0:
                save_checkpoint(
                    ac,
                    ac_targ,
                    pi_optimizer,
                    q_optimizer,
                    epoch,
                    checkpoint_path,
                    gradient_history=gradient_history,
                )

            if epoch % save_freq == 0 or epoch == epochs:
                logger.save_state(
                    {
                        "env": env,
                        "gradient_history": gradient_history,
                    },
                    None,
                )

            test_agent()

            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("ActorGradNorm", average_only=True)
            logger.log_tabular("CriticGradNorm", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)

            logger.dump_tabular()

    save_checkpoint(
        ac,
        ac_targ,
        pi_optimizer,
        q_optimizer,
        epochs,
        checkpoint_path,
        gradient_history=gradient_history,
    )

    return gradient_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=[args.hid] * args.l,
        ),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )