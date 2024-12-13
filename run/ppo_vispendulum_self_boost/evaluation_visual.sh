PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
        --fallback-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum.zip" \
        --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum.zip" \
        --eval-name "fallback_100_agent_100" \
        --log --console --seed 22

PYTHONFAULTHANDLER=1 python pendulum_visual_ppo_load_only.py \
        --fallback-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum.zip" \
        --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
        --eval-name "fallback_100_agent_25" \
        --log --console --seed 22

PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
        --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum.zip" \
        --eval-name "well_trained" \
        --log --console --seed 22

PYTHONFAULTHANDLER=1 python pendulum_visual_ppo.py --notrain \
        --eval-checkpoint "./artifacts/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
        --eval-name "under_trained_25" \
        --log --console --seed 22

