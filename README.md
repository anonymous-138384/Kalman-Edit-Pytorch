### Kalman-Edit Pytorch Implementation
# Environment setup
torch>=2.0.0 diffusers==0.31.0 transformers==4.47.1 numpy==1.24.3
# Demo
CUDA_VISIBLE_DEVICES=0 python demo.py --image ./demo_images/man.png --prompt "A person with glasses" --eta 0.9 --start_timestep 0 --stop_timestep 27 --gamma 0.5
CUDA_VISIBLE_DEVICES=0 python demo.py --image ./demo_images/man.png --prompt "A person with beard" --eta 0.9 --start_timestep 0 --stop_timestep 27 --gamma 0.5
CUDA_VISIBLE_DEVICES=0 python demo.py --image ./demo_images/woman.png --prompt "A person with glasses" --eta 0.9 --start_timestep 0 --stop_timestep 27 --gamma 0.5
CUDA_VISIBLE_DEVICES=0 python demo.py --image ./demo_images/zebra.png --prompt "A zebra stands alone on a vast African savanna with a watering hole in the foreground reflecting zebras and giraffes, and additional zebras and giraffes near acacia trees in the background under the midday sun." --eta 0.9 --start_timestep 0 --stop_timestep 27 --gamma 0.5 --fast True
