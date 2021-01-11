from PIL import Image


def update(size, img_dir, new_img_dir = None):
    im = Image.open(img_dir)
    im_resized = im.resize(size, Image.ANTIALIAS)
    # im_resized.save(new_img_dir, "PNG")
    return im_resized