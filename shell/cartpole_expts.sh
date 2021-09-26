python run.py -m name=pets_cartpole alg=pets env=cartpole seed="range(5)" num_trials=50
python run.py -m name=sac_cartpole alg=sac env=cartpole seed="range(5)" num_epochs=100
python run.py -m name=td3_cartpole alg=td3 env=cartpole seed="range(5)" num_epochs=100
python run.py -m name=ppo_cartpole alg=ppo env=cartpole seed="range(5)" num_epochs=1000
