PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_eval_calf_wrapper.py \
    --fallback-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
    --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
    --eval-name "fallback_25_agent_50" \
    --log --console --seed 22

PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
    --eval-name "agent_50" \
    --log --console --seed 22

PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
    --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
    --eval-name "agent_25" \
    --log --console --seed 22
