from preprocess import *

model_type = 'vit_b'
checkpoint = 'sam_vit_b_01ec64.pth'
device = 'cuda:0'

from segment_anything import SamPredictor, sam_model_registry

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)


# Preprocess the images
from collections import defaultdict

import torch

from segment_anything.utils.transforms import ResizeLongestSide

transformed_data = defaultdict(dict)
for k in tqdm(bbox_coords.keys()):
    image = images[k]
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image)
    transformed_image = input_image_torch.permute(
        2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size

# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-5
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(),
                             lr=lr,
                             weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())

from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

num_epochs = 10
losses = []
sam_model.to(device)
sam_model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    print("training epoch {0} ...".format(epoch))
    # Just train on the first 20 examples
    for k in tqdm(keys[:train_size]):
        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)

            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(
            low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        gt_mask_resized = torch.from_numpy(
            np.resize(ground_truth_masks[k],
                      (1, 1, ground_truth_masks[k].shape[0],
                       ground_truth_masks[k].shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0,
                                         dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

mean_losses = [mean(x) for x in losses]
mean_losses

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.savefig("loss.png")
#plt.show()


# Load up the model with default weights
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device)

# Set up predictors for both tuned and original models
from segment_anything import sam_model_registry, SamPredictor

predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)

# The model has not seen keys[21] (or keys[20]) since we only trained on keys[:20]
k = keys[train_size+100]
image = images[k]

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array(bbox_coords[k])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

#%matplotlib inline
_, axs = plt.subplots(1, 2, figsize=(25, 25))

axs[0].imshow(image)
show_mask(masks_tuned, axs[0])
show_box(input_bbox, axs[0])
axs[0].set_title('Mask with Tuned Model', fontsize=26)
axs[0].axis('off')

axs[1].imshow(image)
show_mask(masks_orig, axs[1])
show_box(input_bbox, axs[1])
axs[1].set_title('Mask with Untuned Model', fontsize=26)
axs[1].axis('off')

plt.savefig("compare.png")
#plt.show()

sam_model.eval()
torch.save(sam_model,'fine_tuned_sam_4_tn_scui.pth')