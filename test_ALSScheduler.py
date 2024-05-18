import numpy as np
import PIL.Image
import torch
import tqdm
from diffusers import ALSScheduler, UNet2DModel


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.detach().numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    print(f"Image at step {i}")
    image_pil.show()
    image_pil.save("result.png")


repo_id = "ffxxxx/sd-class-butterflies-32"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
#scheduler = DDPMScheduler.from_pretrained(repo_id)
scheduler = ALSScheduler.from_pretrained(repo_id)
without_loop=scheduler.whether_without_loop()

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)

sample = noisy_sample

if without_loop:
    print("without_loop")
    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            model_output = model(sample, t).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        sample = scheduler.step_without_loop(model_output, t, sample).prev_sample

        # 3. optionally look at image
        if (i + 1) in [900,1000]:
            display_sample(sample, i + 1)

else:
    # Annealed Langevin Sampling
    pass
    print("with_loop")
    # i=scheduler.__len__()-1
    epsilon=1e-5
    loop_times=1
    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        step_size=epsilon*((scheduler.get_sigma(int(t)))**2)/((scheduler.get_sigma(0))**2)

        for _ in range(loop_times):
            # get the model_output
            with torch.no_grad():
                model_output=model(sample,t).sample
            # update the sample
            sample=scheduler.step_langevin(model_output*2, t, sample, step_size).prev_sample
        
        if i in [800,900,999]:
            #optionally show pics
            display_sample(sample, i + 1)




    # print("with_loop")
    # #i=scheduler.__len__()-1
    # i=900
    # epsilon=1e-5
    # while(i>=0):
    #     step_size=epsilon*((scheduler.get_sigma(i))**2)/((scheduler.get_sigma(0))**2)

    #     for j, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    #         # get the model_output
    #         with torch.no_grad():
    #             model_output=model(sample,t).sample

    #         # update the sample
    #         sample=scheduler.step_langevin(2*model_output, t, sample, step_size).prev_sample

    #     i=i-100
    #     if i<=250:
    #         display_sample(sample, i + 1)
        # if i>=200:
        #     i=i-100

        # else:
        #     i=i-50
        #     display_sample(sample, i + 1)

        # optionally show pics
        # if (i + 1) %2==0:
        #     display_sample(sample, i + 1)
        



# image_processed = sample.cpu().permute(0, 2, 3, 1)
# image_processed = (image_processed + 1.0) * 127.5
# image_processed = image_processed.numpy().astype(np.uint8)
# image_pil = PIL.Image.fromarray(image_processed[0])
# image_pil.show()
# image_pil.save("result.png")
# for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):

