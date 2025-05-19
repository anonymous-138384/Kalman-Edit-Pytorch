import argparse
import torch
from pipeline_kalman_edit import FluxKalmanPipeline
from pipeline_kalman_edit_fast import FluxFastKalmanPipeline
from diffusers.utils import load_image
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Run Flux Kalman Edit Pipeline")
    parser.add_argument("--model", type=str, default="/root/flux/", help="FLUX model path")
    parser.add_argument("--fast", type=bool, default=False, help="If use Kalman-Edit or Kalman-Edit* algorithm")
    parser.add_argument("--image", type=str, default="./demo_images/man.png", help="Path of the input image")
    parser.add_argument("--prompt", type=str, default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k", help="Prompt for image generation")
    parser.add_argument("--prompt_2", type=str, help="Second prompt (if different from prompt)")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength parameter")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter")
    parser.add_argument("--eta", type=float, default=0.9, help="Eta parameter")
    parser.add_argument("--start_timestep", type=int, default=0, help="Start timestep")
    parser.add_argument("--stop_timestep", type=int, default=2, help="Stop timestep")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image filename")
    parser.add_argument("--use_img2img", action="store_true", help="Use FluxImg2ImgPipeline")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--mu", type=float, default=0.7, help="Kalman control parameter")
    parser.add_argument("--lambda1", type=float, default=0.1, help="Kalman control parameter")
    parser.add_argument("--semantic_param_stage1", type=int, default=3, help="end step of semantic formed in stage 1")
    parser.add_argument("--semantic_param_stage2", type=int, default=10, help="end step of semantic formed in stage 2")
    parser.add_argument("--refine_param_stage1", type=int, default=14, help="start step of refinement in stage 1")
    parser.add_argument("--refine_param_stage2", type=int, default=15, help="start step of refinement in stage 2")
    parser.add_argument("--generation_strength_stage1", type=float, default=0.35, help="controller strength in stage 1")
    parser.add_argument("--delta", type=int, default=15, help="parameter for constructing measurement sequence")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fast:
        pipe = FluxFastKalmanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxKalmanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    if args.image.startswith('http://') or args.image.startswith('https://'):
        init_image = load_image(args.image).resize((1024, 1024))
    else:
        init_image = Image.open(args.image).resize((1024, 1024))

    prompt_2 = args.prompt_2 if args.prompt_2 else args.prompt
    save_base, ext = args.output.rsplit(".", 1)
    for i in range(args.num_images):
        kwargs = {"gamma": args.gamma, "eta": args.eta, "start_timestep": args.start_timestep, "stop_timestep": args.stop_timestep} if not args.use_img2img else dict({})
        images = pipe(
            prompt=args.prompt,
            prompt_2=prompt_2,
            image=init_image,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            img_path=args.image,
            mu1=args.mu,
            lambda1=args.lambda1,
            semantic_param_stage1=args.semantic_param_stage1,
            semantic_param_stage2=args.semantic_param_stage2,
            refine_param_stage1=args.refine_param_stage1,
            refine_param_stage2=args.refine_param_stage2,
            generation_strength_stage1=args.generation_strength_stage1,
            delta=args.delta,
            **kwargs,
        ).images[0]
        #images = images.resize((1360,768))
        images.save(f"./out.png")
        print(f"Output image saved ")

if __name__ == "__main__":
    main()