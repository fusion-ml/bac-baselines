# python pets.py -m name=pets_cartpole env=cartpole seed="range(5)" num_trials=50
# python sac.py -m name=sac_cartpole alg=sac env=cartpole seed="range(5)" num_epochs=100
python td3.py -m name=td3_cartpole alg=td3 env=cartpole seed="range(5)" num_epochs=100
python ppo.py -m name=ppo_cartpole alg=ppo env=cartpole seed="range(5)" num_epochs=1000
