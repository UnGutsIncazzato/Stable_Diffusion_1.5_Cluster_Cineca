from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
import random
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import lpips
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #Use GPU if available
STORY_FILE = 'story.txt' #Chose the file containing the story prompts, in this case the sd8.txt is located in the same directory as the script
STYLE_PROMPT = ', cinematic photorealism, ultra detailed, sharp focus, coherent style, 8k, trending on artstation' #Prompt to add style to each image
OUTPUT_DIR = 'generated_image' #Output directory to save generated images and metrics
BASE_SEED = 120 #You can chose any base seed you want
NUM_IMAGES_TO_GENERATE = 5 #Number of images to generate based on the story prompts

#Context note: you can add fixed context to all prompts here, you can even describe without it
FIXED_CONTEXT='' #Example:' same character,ancient wizard, long grey beard, pointed hat, same environment, mossy tower ruin, medieval setting,'

# Numeber of previous prompts to include as context for few-shot learning (0 = no context), i sugesgest to keep it to 0 for this sd8 story, but no more than 3
CONTEXT_HISTORY_SIZE = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function of segmenting story text into coherent parts
def segment_story_text(story_text, num_segments):
    segments = [s.strip() for s in story_text.strip().split('\n\n') if s.strip()]
    return segments[:num_segments]


#Function to get dynamic strength based on frame index
def get_dynamic_strength(frame_index, total_frames):
    
    
    if frame_index == 0: return 0.0 # T2I this is the first frame generated from text
    if frame_index == 1: return 0.95 # Slightly higher strength for the second frame, cause with a lower strength the model may ignore new elements and don't change
    if frame_index == 2: return 0.90 
    if frame_index == 3: return 0.85
    return 0.85 



#prompts_to_use="sd8.txt"
prompts_to_use = [
    "A solitary, ancient wizard with a long grey beard reads a huge, open book, sitting on a mossy stone slab outside a ruined tower.",
    "The wizard raises his left hand, and a spell manifests, the stone slab to his right is empty, but the magic illuminates the surroundings with a blue glow.",
    "(A sleek, black cat suddenly jumps onto the stone slab to the wizard's right:1.45), detailed black fur, glowing eyes, startling the wizard.", # <--- ANCORAGGIO DEL GATTO + PESO ESTREMO 1.45
    "The wizard, startled by the cat's presence, lowers his hand, and the blue spell dissipates, looking gently at the cat sitting on the stone slab to his right.",
    "With a gentle sigh, the wizard carefully puts his book down on the stone slab next to the cat, reaching out a hand to gently pet the black cat."
]

TOTAL_COUNT = len(prompts_to_use)
print(f"Lecture complete of the story from: {STORY_FILE}...")
try:
    with open(STORY_FILE, 'r') as f:
        full_story_content = f.read()
except FileNotFoundError:
    print(f"Errore: File della storia non trovato: {STORY_FILE}")
    exit()

prompts_to_use = segment_story_text(full_story_content, NUM_IMAGES_TO_GENERATE)
TOTAL_COUNT = len(prompts_to_use)
print(f"Story segmented in {TOTAL_COUNT} unique prompt to use. Starting...")


# Stable Diffusion Pipeline (Text-to-Image)
print("\nLoading Stable Diffusion T2I (local)...")
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    local_files_only=True
).to(DEVICE)

# Pipeline Img2Img for coerent image sequences
print("\nLoading Stable Diffusion I2I (local)...")
pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    local_files_only=True
).to(DEVICE)

# VAE Refinement
print("\nLoading VAE Refinement (stabilityai/sd-vae-ft-mse) in local mode...")
try:
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(DEVICE)
    pipe_t2i.vae = vae
    pipe_i2i.vae = vae
    print(" Vae applied to both pipeline (local)...")
except Exception as e:
    print(f"WARNING: Impossible to load the VAE localy: {e}")
    print("The pipeline will continue without VAE Refinement. Make sure the model is in the cache.")


print("\nConfiguration of Scheduler: DPM Solver++ 2M Karras...")
scheduler = DPMSolverMultistepScheduler.from_config(pipe_t2i.scheduler.config, use_karras_sigmas=True)
pipe_t2i.scheduler = scheduler
pipe_i2i.scheduler = scheduler


# Configuration of LPIPS e CLIP models for metrics calculation
print(f"\nCaricamento modello LPIPS (AlexNet) su {DEVICE}...")
try:
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
except Exception as e:
    loss_fn_lpips = lpips.LPIPS(net='alex', pretrained=False).to(DEVICE)
loss_fn_lpips.eval()

# CLIP Model
model_name_clip = "openai/clip-vit-base-patch32"
print(f"Caricamento modello CLIP (forzato da cache) su {DEVICE}...")
model_clip = CLIPModel.from_pretrained(
    model_name_clip,
    local_files_only=True
).to(DEVICE)
processor_clip = CLIPProcessor.from_pretrained(
    model_name_clip,
    local_files_only=True
)

# Load image and preprocess for LPIPS
def load_image_tensor_lpips(path):
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        return None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

#calculate LPIPS coherence between consecutive images
def calculate_lpips_coherence(image_paths, lpips_fn):
    if len(image_paths) < 2: return 0.0, []
    all_distances = []
    log_details = []
    img_prev = load_image_tensor_lpips(image_paths[0])
    if img_prev is None: return 0.0, []
    print("\nStarting Calculating LPIPS (Visibility Coerence)...")
    for i in tqdm(range(len(image_paths) - 1), desc="Calculate LPIPS"):
        img_curr = load_image_tensor_lpips(image_paths[i+1])
        if img_curr is None: continue
        with torch.no_grad():
            dist = lpips_fn(img_prev, img_curr).item()
            all_distances.append(dist)
            log_details.append((os.path.basename(image_paths[i]), os.path.basename(image_paths[i+1]), dist))
        img_prev = img_curr
    if all_distances:
        return sum(all_distances) / len(all_distances), log_details
    return 0.0, []

# Calculate CLIP score for single image and text prompt
def calculate_clip_score_single_image(image_path, text_prompt):
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        return None
    inputs = processor_clip(
        text=text_prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    if 'position_ids' not in inputs:
        input_ids_len = inputs['input_ids'].shape[1]
        inputs['position_ids'] = torch.arange(input_ids_len).unsqueeze(0)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_clip(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    clip_score = torch.sum(image_embeds * text_embeds)
    return clip_score.item()



# Initialization before generation loop
# Negative prompt string to avoid unwanted elements
negative_prompt_string = "text, writing, words, signature, watermark, logo, cropped, text, writing, words, signature, watermark, logo, cropped, blurry, distorted, low quality, worst quality, low resolution, noise, jpeg artifacts, grain, ugly, oversaturated, disfigured, deformed, mutation, extra limbs, missing limbs, weird eyes, inconsistent face, wrong proportions, poorly drawn face, bad anatomy, malformed, out of frame, extra fingers, too many hands, duplicate person, tiling, cartoon, sketch, painting, illustration, drawing, plastic, toy, figurine, sculpture, fake, artificial, dry skin, dull water, flat light, monochrome"

generated_image_paths = [] 
story_context_history = []
img_prev = None

# Variables for generation loop
GUIDANCE_SCALE = 15.0 # Increased guidance scale for stronger adherence to prompts Suggested to be between 7.5 and 15
NUM_INFERENCE_STEPS = 150 # Usually 50-100 steps are enough, but for I2I with complex prompts we can use more steps

print(f"\Start creating of {TOTAL_COUNT} images sequentially with T2I/I2I...")


for i, prompt in enumerate(prompts_to_use):

    # Logic for context history in few-shot learning
    context_history = story_context_history[-CONTEXT_HISTORY_SIZE:]
    context_prefix = ", ".join(context_history)

    # Prompt weighting for narrative elements
    weighted_prompt_narrative = f"({prompt}:1.35)" # Weighted bewteen 1.2 and 1.5 for main narrative elements

    # Compose final prompt with context history if available
    if i > 0:
        prompt_segment = f"Recent context: {context_prefix}. Current scene: {weighted_prompt_narrative}" # If not the first frame, add context history
    else:
        prompt_segment = prompt 

    full_prompt = FIXED_CONTEXT + prompt_segment + STYLE_PROMPT
    story_context_history.append(prompt) 
    current_seed = BASE_SEED + random.randint(0, 5) # Slightly vary seed for each frame
    generator = torch.Generator(DEVICE).manual_seed(current_seed)
    current_strength = get_dynamic_strength(i, TOTAL_COUNT)

    # LOGIC FOR T2I/I2I GENERATION
    if i == 0:
        print(f"Generate the first image with Text-to-Image (Frame {i:02d}) with weighted prompt. Strength: N/A")
        image = pipe_t2i(
            prompt=full_prompt,
            negative_prompt=negative_prompt_string,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=768,
            width=768,
            generator=generator
        ).images[0]
    else:
        print(f"Generate an image with Image-to-Image (Frame {i:02d}) with weighted prompt. Strength: {current_strength:.2f}")
        image = pipe_i2i(
            prompt=full_prompt,
            image=img_prev,
            strength=current_strength, 
            negative_prompt=negative_prompt_string,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS, 
            generator=generator
        ).images[0]

    # Save the generated image on the output directory estibilished before
    output_path = os.path.join(OUTPUT_DIR, f"image_{i:02d}.png")
    image.save(output_path)
    generated_image_paths.append(output_path)
    img_prev = image

print("\nEnd of generation loop!")



# Calulate CLIP score for each image and prompt and save results in txt file
all_clip_scores = []
clip_output_lines = []
CLIP_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clip_score_details.txt")

print("Start calculating CLIP scores for each image and prompt..")
for i, image_path in enumerate(generated_image_paths):
    prompt_di_valutazione = FIXED_CONTEXT + prompts_to_use[i] + STYLE_PROMPT
    score = calculate_clip_score_single_image(image_path, prompt_di_valutazione)

    if score is not None:
        all_clip_scores.append(score)
        prompt_visualizzazione = prompts_to_use[i]
        log_line = f"[{i+1}/{TOTAL_COUNT}] File: {os.path.basename(image_path)} | Prompt: \"{prompt_visualizzazione[:40]}...\" | CLIP Score: {score:.4f}"
        clip_output_lines.append(log_line)

if all_clip_scores:
    avg_clip_score = sum(all_clip_scores) / len(all_clip_scores)
    final_clip_summary = "\n" + "="*70 + f"\n AVERAGE CLIP FINAL SCORE (Textual Coeherence): {avg_clip_score:.4f} (Base on {len(all_clip_scores)} results)" + "\n" + "="*70
    clip_output_lines.append(final_clip_summary)
   

    with open(CLIP_OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(clip_output_lines))
    print(final_clip_summary)
    print(f"Clip results saved successfully in: {CLIP_OUTPUT_FILE}")
    
else:
    print(f"WARING: The list 'all_clip_scores' is empty ({len(all_clip_scores)}), no CLIP output file written. Output file not written.")


lpips_score, lpips_log_details = calculate_lpips_coherence(generated_image_paths, loss_fn_lpips)
lpips_output_lines = []
LPIPS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lpips_coherence_details.txt")

if lpips_log_details:
    for i, (img1, img2, dist) in enumerate(lpips_log_details):
        log_line = f"[Transiction {i+1:02d}] {img1} -> {img2} | LPIPS Distance: {dist:.4f}"
        lpips_output_lines.append(log_line)

     
    final_lpips_summary = "\n" + "="*70 + f"\nDISTANCE LPIPS AVERAGE (Visual Coherence ): {lpips_score:.4f} (Based on {len(lpips_log_details)} transition)" + "\n" + "="*70
    lpips_output_lines.append(final_lpips_summary)

    with open(LPIPS_OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lpips_output_lines))
    print(final_lpips_summary)
    print(f"LPIPS Results Saved in: {LPIPS_OUTPUT_FILE}")
else:
    print(f" WARNING: The list 'lpips_log_details' is empty ({len(lpips_log_details)} transition). Output file not written.")

print("\n--- ALL DONE: METRICS CREATED (CLIP & LPIPS) ---\n")
