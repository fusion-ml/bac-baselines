python pets.py -m name=pets_swimmer env=swimmer_pets seed="range(5)" num_trials=20
python sac.py -m name=sac_swimmer alg=sac env=swimmer seed="range(5)" num_epochs=50
python td3.py -m name=td3_swimmer alg=td3 env=swimmer seed="range(5)" num_epochs=50
python ppo.py -m name=ppo_swimmer alg=ppo env=swimmer seed="range(5)" num_epochs=50
