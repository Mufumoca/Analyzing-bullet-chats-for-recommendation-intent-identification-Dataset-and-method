from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PAGE_DIR = ROOT / "pages_final"
OUTPUT = ROOT / "report_contact_sheet.png"
THUMB_WIDTH = 300
COLUMNS = 4
MARGIN = 18
LABEL_HEIGHT = 28


def main() -> None:
    page_paths = sorted(PAGE_DIR.glob("page-*.png"))
    if not page_paths:
        raise FileNotFoundError(f"No rendered pages in {PAGE_DIR}")
    thumbnails: list[Image.Image] = []
    for path in page_paths:
        with Image.open(path) as image:
            ratio = THUMB_WIDTH / image.width
            resized = image.convert("RGB").resize(
                (THUMB_WIDTH, int(image.height * ratio)), Image.Resampling.LANCZOS
            )
            thumbnails.append(resized)
    cell_height = max(image.height for image in thumbnails) + LABEL_HEIGHT
    rows = (len(thumbnails) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB",
        (
            MARGIN + COLUMNS * (THUMB_WIDTH + MARGIN),
            MARGIN + rows * (cell_height + MARGIN),
        ),
        "#D9D9D9",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, image in enumerate(thumbnails):
        row, column = divmod(index, COLUMNS)
        x = MARGIN + column * (THUMB_WIDTH + MARGIN)
        y = MARGIN + row * (cell_height + MARGIN)
        sheet.paste(image, (x, y + LABEL_HEIGHT))
        draw.text((x + 4, y + 4), f"Page {index + 1}", fill="#222222", font=font)
    sheet.save(OUTPUT, dpi=(150, 150))
    print(f"Rendered {len(thumbnails)} pages into {OUTPUT}")


if __name__ == "__main__":
    main()
