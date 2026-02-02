import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
from pathlib import Path


class HTML:
    """Small helper to write an image gallery HTML page using dominate."""

    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = Path(web_dir)
        self.img_dir = self.web_dir / "images"

        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, text):
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=Path("images") / link):
                                img(style=f"width:{width}px", src=Path("images") / im)
                            br()
                            p(txt)

    def save(self):
        html_file = self.web_dir / "index.html"
        with open(html_file, "wt") as f:
            f.write(self.doc.render())
