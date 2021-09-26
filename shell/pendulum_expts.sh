python run.py -m name=pets_pendulum alg=pets env=pendulum seed="range(5)" num_trials=50
python run.py -m name=sac_pendulum alg=sac env=pendulum seed="range(5)" num_epochs=100
python run.py -m name=td3_pendulum alg=td3 env=pendulum seed="range(5)" num_epochs=100
python run.py -m name=ppo_pendulum alg=ppo env=pendulum seed="range(5)" num_epochs=100
