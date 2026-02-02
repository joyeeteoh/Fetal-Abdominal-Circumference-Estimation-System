from pathlib import Path

from . import util


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to disk and add them to an HTML page."""
    image_dir = webpage.get_image_dir()
    name = Path(image_path[0]).stem

    webpage.add_header(name)

    ims, txts, links = [], [], []
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = f"{name}_{label}.png"
        save_path = image_dir / image_name
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)

    webpage.add_images(ims, txts, links, width=width)
