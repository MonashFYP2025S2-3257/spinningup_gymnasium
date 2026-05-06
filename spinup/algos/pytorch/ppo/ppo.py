import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.tools import statistics_scalar


def save_checkpoint(ac, pi_optimizer, vf_optimizer, epoch, path, gradient_history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "actor_critic": ac.state_dict(),
        "pi_opt": pi_optimizer.state_dict(),
        "vf_opt": vf_optimizer.state_dict(),
        "epoch": epoch,
        "gradient_history": gradient_history,
    }, path)


def load_checkpoint(ac, pi_optimizer, vf_optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    ac.load_state_dict(checkpoint["actor_critic"])
    pi_optimizer.load_state_dict(checkpoint["pi_opt"])
    vf_optimizer.load_state_dict(checkpoint["vf_opt"])
    return checkpoint["epoch"], checkpoint.get("gradient_history", None)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        if adv_std == 0:
            adv_std = 1.0
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # -------------------- checkpoint setup --------------------
    exp_name = logger_kwargs.get("exp_name", "ppo_run")
    checkpoint_root = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_root, f"{exp_name}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # -------------------- layerwise gradient history --------------------
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
            summary[k] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        return summary

    def reset_epoch_gradient_buffer():
        epoch_grad_buffer.clear()

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    logger.setup_pytorch_saver(ac)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Found {checkpoint_path}")
        try:
            start_epoch, loaded_history = load_checkpoint(ac, pi_optimizer, vf_optimizer, checkpoint_path)
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

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        last_loss_pi = pi_l_old
        last_loss_v = v_l_old
        last_pi_info = pi_info_old
        stop_iter = 0

        # -------- Policy update --------
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)

            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                stop_iter = i
                last_pi_info = pi_info
                last_loss_pi = loss_pi.item()
                break

            loss_pi.backward()

            actor_gn = 0.0
            for p in ac.pi.parameters():
                if p.grad is not None:
                    actor_gn += p.grad.data.norm(2).item() ** 2
            actor_gn = actor_gn ** 0.5

            pi_grads = get_layer_grad_norms(ac.pi, "actor")
            accumulate_gradients(pi_grads)

            pi_optimizer.step()

            logger.store(ActorGradNorm=actor_gn)
            last_pi_info = pi_info
            last_loss_pi = loss_pi.item()
            stop_iter = i + 1

        # -------- Value update --------
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()

            critic_gn = 0.0
            for p in ac.v.parameters():
                if p.grad is not None:
                    critic_gn += p.grad.data.norm(2).item() ** 2
            critic_gn = critic_gn ** 0.5

            v_grads = get_layer_grad_norms(ac.v, "critic")
            accumulate_gradients(v_grads)

            vf_optimizer.step()

            logger.store(CriticGradNorm=critic_gn)
            last_loss_v = loss_v.item()

        kl, ent, cf = last_pi_info['kl'], pi_info_old['ent'], last_pi_info['cf']
        logger.store(
            StopIter=stop_iter,
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(last_loss_pi - pi_l_old),
            DeltaLossV=(last_loss_v - v_l_old),
        )

    start_time = time.time()
    o, ep_ret, ep_len = env.reset()[0], 0, 0

    reset_epoch_gradient_buffer()

    for epoch in range(start_epoch, epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated

            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps.', flush=True)

                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0

                buf.finish_path(v)

                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, ep_ret, ep_len = env.reset()[0], 0, 0

        # Perform PPO update
        update()

        # Save mean layerwise gradients for this epoch
        gradient_history[epoch + 1] = summarize_epoch_gradients()
        reset_epoch_gradient_buffer()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                ac,
                pi_optimizer,
                vf_optimizer,
                epoch + 1,
                checkpoint_path,
                gradient_history=gradient_history,
            )

        # Save spinup state
        if ((epoch + 1) % save_freq == 0) or ((epoch + 1) == epochs):
            logger.save_state({"env": env, "gradient_history": gradient_history}, None)

        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('ActorGradNorm', average_only=True)
        logger.log_tabular('CriticGradNorm', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    return gradient_history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )