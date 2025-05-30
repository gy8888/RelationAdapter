import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import os
import numpy as np
from transformers import AutoProcessor
import torchvision.transforms.functional as F

Image.MAX_IMAGE_PIXELS = None

def make_train_dataset(args, tokenizer, accelerator=None):
    if args.train_data_dir is not None:
        print("loading dataset ... ")
        dataset = load_dataset('json', data_files=args.train_data_dir)
        base_path = os.path.dirname(os.path.abspath(args.train_data_dir))

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.caption_column is None:
        caption_column = column_names[0]
        print(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.source_column is None:
        source_column = column_names[1]
        print(f"source column defaulting to {source_column}")
    else:
        source_column = args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"`--source_column` value '{args.source_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.target_column is None:
        target_column = column_names[2]
        print(f"target column defaulting to {target_column}")
    else:
        target_column = args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"`--target_column` value '{args.target_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    #Add two images as columnname
    if args.ipa_source_column is None:
        ipa_source_column = column_names[3]
        print(f"ipa-source column defaulting to {ipa_source_column}")
    else:
        ipa_source_column = args.ipa_source_column
        if ipa_source_column not in column_names:
            raise ValueError(
                f"`--ipa_source_column` value '{args.ipa_source_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.ipa_target_column is None:
        ipa_target_column = column_names[4]
        print(f"ipa-target column defaulting to {ipa_target_column}")
    else:
        ipa_target_column = args.ipa_target_column
        if ipa_target_column not in column_names:
            raise ValueError(
                f"`--ipa_target_column` value '{args.ipa_target_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def resize_long_side(img, target_long_side, interpolation=transforms.InterpolationMode.BILINEAR):
        w, h = img.size  
        if w >= h:
            new_w = target_long_side
            new_h = int(target_long_side * h / w)
        else:
            new_h = target_long_side
            new_w = int(target_long_side * w / h)
        return F.resize(img, (new_h, new_w), interpolation=interpolation)

    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_long_side(img, args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    def tokenize_prompt_clip_t5(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)                    
            elif isinstance(caption, list):
                captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        _examples = {}        

        source_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[source_column]]
        target_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[target_column]]

        _examples["cond_pixel_values"] = [train_transforms(source) for source in source_images]
        _examples["pixel_values"] = [train_transforms(image) for image in target_images]
        _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples)

        # clip pre-processor
        # ————————————————————————————————————————————————————————
        clip_image_processor = AutoProcessor.from_pretrained('../models/siglip-so400m-patch14-384')
        ipa_source_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[ipa_source_column]]
        ipa_target_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[ipa_target_column]]
        _examples["ipa_source_images"] = [clip_image_processor(images=source, return_tensors="pt").pixel_values for source in ipa_source_images]
        _examples["ipa_target_images"] = [clip_image_processor(images=image, return_tensors="pt").pixel_values for image in ipa_target_images]
        
        drop_image_embeds = [1 if random.random() < 0.05 else 0 for _ in examples[ipa_target_column]]
        _examples["drop_image_embeds"] = drop_image_embeds
        # print(f"ipa_source_images[0] shape: {_examples['ipa_source_images'][0].shape}")
        # print(f"ipa_target_images[0] shape: {_examples['ipa_target_images'][0].shape}")

        # —————————————————————————————————————————————————————————
        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    # token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    # token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])
    
    token_ids_clip = torch.stack([
        example["token_ids_clip"].clone().detach() if isinstance(example["token_ids_clip"], torch.Tensor)
        else torch.tensor(example["token_ids_clip"])
        for example in examples
    ])
    token_ids_t5 = torch.stack([
        example["token_ids_t5"].clone().detach() if isinstance(example["token_ids_t5"], torch.Tensor)
        else torch.tensor(example["token_ids_t5"])
        for example in examples
    ])

    # ————————————————————————————————————————————————————————————————————————————
    ipa_source_pixel_values = torch.cat([example["ipa_source_images"] for example in examples])
    ipa_source_pixel_values = ipa_source_pixel_values.to(memory_format=torch.contiguous_format).float()
    ipa_target_pixel_values = torch.cat([example["ipa_target_images"] for example in examples])
    ipa_target_pixel_values = ipa_target_pixel_values.to(memory_format=torch.contiguous_format).float()
    drop_image_embeds = [example["drop_image_embeds"] for example in examples]
    return {
        "cond_pixel_values": cond_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
        "ipa_source_pixel_values": ipa_source_pixel_values,
        "ipa_target_pixel_values": ipa_target_pixel_values,
        "drop_image_embeds": drop_image_embeds
    }
    # ————————————————————————————————————————————————————————————————————————————