import os
import json
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from natsort import natsorted

def combine_images_vertically(img1_path, img2_path):
    im1 = Image.open(img1_path)
    im2 = Image.open(img2_path)
    w1, h1 = im1.size
    w2, h2 = im2.size
    new_im = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (0, h1))
    return new_im

def process_subsubfolder(folder_path, input_root, output_root, output_test_root):
    labels_path = os.path.join(folder_path, "labels.json")
    if not os.path.exists(labels_path):
        return
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = [data] if isinstance(data, dict) else data
    label_dict = {e['img_name']: e for e in entries}

    exts = {'.png','.jpg','.jpeg','.bmp','.gif'}
    imgs = [f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in exts and f in label_dict]
    imgs = natsorted(imgs)
    n_total = len(imgs)
    if n_total < 2:
        return

    rel = os.path.relpath(folder_path, input_root)
    out1 = os.path.join(output_root, rel)
    out2 = os.path.join(output_test_root, rel)
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    if n_total > 19:
        core_imgs = imgs[:-5]
        reserve_imgs = imgs[-5:]
    else:
        core_imgs = imgs
        reserve_imgs = []

    limit = 2000
    existing = len([f for f in os.listdir(out1) if f.endswith('.png')])
    new_cnt = 0
    m = len(core_imgs)
    total1 = m * (m - 1)
    pbar1 = tqdm(total=total1, desc=f"[Phase1] {os.path.basename(folder_path)}", leave=False)
    for offset in range(1, m):
        for i in range(m):
            if existing + new_cnt >= limit:
                break
            img1 = core_imgs[i]
            img2 = core_imgs[(i + offset) % m]
            idx = existing + new_cnt + 1
            name_png = f"{idx:06d}.png"
            name_txt = f"{idx:06d}.txt"
            path_png = os.path.join(out1, name_png)
            path_txt = os.path.join(out1, name_txt)

            if os.path.exists(path_png) and os.path.exists(path_txt):
                new_cnt += 1
                pbar1.update(1)
                continue

            comb = combine_images_vertically(
                os.path.join(folder_path, img1),
                os.path.join(folder_path, img2)
            )
            comb.save(path_png)
            inst = label_dict[img2].get("edit_instruction", "")
            with open(path_txt, 'w', encoding='utf-8') as fw:
                fw.write(inst)

            new_cnt += 1
            pbar1.update(1)
        else:
            continue
        break
    pbar1.close()

    if reserve_imgs:
        k = len(reserve_imgs)
        total2 = (n_total - k) * k + k * (k - 1)
        existing2 = len([f for f in os.listdir(out2) if f.endswith('.png')])
        new2 = 0
        pbar2 = tqdm(total=total2, desc=f"[Phase2] {os.path.basename(folder_path)}", leave=False)

        for img1 in core_imgs:
            for img2 in reserve_imgs:
                idx2 = existing2 + new2 + 1
                name2_png = f"{idx2:06d}.png"
                name2_txt = f"{idx2:06d}.txt"
                path2_png = os.path.join(out2, name2_png)
                path2_txt = os.path.join(out2, name2_txt)

                if os.path.exists(path2_png) and os.path.exists(path2_txt):
                    new2 += 1
                    pbar2.update(1)
                    continue

                comb2 = combine_images_vertically(
                    os.path.join(folder_path, img1),
                    os.path.join(folder_path, img2)
                )
                comb2.save(path2_png)
                inst2 = label_dict[img2].get("edit_instruction", "")
                with open(path2_txt, 'w', encoding='utf-8') as fw2:
                    fw2.write(inst2)

                new2 += 1
                pbar2.update(1)

        for i, img1 in enumerate(reserve_imgs):
            for img2 in reserve_imgs:
                if img1 == img2:
                    continue
                idx2 = existing2 + new2 + 1
                name2_png = f"{idx2:06d}.png"
                name2_txt = f"{idx2:06d}.txt"
                path2_png = os.path.join(out2, name2_png)
                path2_txt = os.path.join(out2, name2_txt)

                if os.path.exists(path2_png) and os.path.exists(path2_txt):
                    new2 += 1
                    pbar2.update(1)
                    continue

                comb2 = combine_images_vertically(
                    os.path.join(folder_path, img1),
                    os.path.join(folder_path, img2)
                )
                comb2.save(path2_png)
                inst2 = label_dict[img2].get("edit_instruction", "")
                with open(path2_txt, 'w', encoding='utf-8') as fw2:
                    fw2.write(inst2)

                new2 += 1
                pbar2.update(1)

        pbar2.close()

def process_all(input_root, output_root, output_test_root):
    subsub = []
    for d1 in os.listdir(input_root):
        p1 = os.path.join(input_root, d1)
        if not os.path.isdir(p1): continue
        for d2 in os.listdir(p1):
            p2 = os.path.join(p1, d2)
            if os.path.isdir(p2):
                subsub.append(p2)

    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as execu:
        list(tqdm(execu.map(
            lambda fp: process_subsubfolder(fp, input_root, output_root, output_test_root),
            subsub
        ), total=len(subsub), desc="Processing folders"))

if __name__ == "__main__":
    input_directory      = "Data_seen"
    output_directory     = "Dataset-Seen-2000-train"
    output_directory_test= "Dataset-Seen-2000-test"
    process_all(input_directory, output_directory, output_directory_test)