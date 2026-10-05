"""Readable camera evidence, with callouts outside detected people."""
from PIL import Image, ImageDraw, ImageFont

from .evidence_layout import layout_labels


def render_observation(observation: dict, source: Image.Image | None = None) -> Image.Image:
    width, height = observation["width"], observation["height"]
    annotated = source is not None
    if source is None:
        scale = min(1.0, 960 / max(width, height))
        source = Image.new("RGB", (max(1, round(width * scale)), max(1, round(height * scale))), (46, 35, 26))
    else:
        # Render text at a useful display size without claiming new camera detail.
        scale = min(2.0, 1280 / source.width, 1600 / source.height)
        if scale != 1:
            source = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
    image_width, image_height = source.size
    sx, sy = image_width / width, image_height / height
    unit = max(.65, min(1.25, image_width / 1280))
    font = ImageFont.load_default(size=round(19 * unit))
    small_font = ImageFont.load_default(size=round(17 * unit))
    count_font = ImageFont.load_default(size=round(30 * unit))
    caption_font = ImageFont.load_default(size=round(23 * unit))
    padding_x, padding_y = round(11 * unit), round(7 * unit)
    rows, person_number = [], 0
    # Frame-local spatial order agrees with the previous evidence numbering.
    for detection in sorted(observation["detections"], key=lambda d: (d["label"] == "person", d["xyxy"][1], d["xyxy"][0])):
        coords = detection["xyxy"]
        rect = (max(0, min(image_width - 1, round(coords[0] * sx))),
                max(0, min(image_height - 1, round(coords[1] * sy))),
                max(0, min(image_width - 1, round(coords[2] * sx))),
                max(0, min(image_height - 1, round(coords[3] * sy))))
        if detection["label"] == "person":
            if detection.get("includedInCount"):
                person_number += 1
                label, color = f"Person {person_number}", "#45DBA4"
            else:
                label, color = "Outside count area", "#F5B857"
        else:
            label, color = detection["label"], "#B6C9EF"
        bounds = font.getbbox(label)
        rows.append({"rect": rect, "label": label, "color": color,
                     "size": (bounds[2] - bounds[0] + 2 * padding_x,
                              bounds[3] - bounds[1] + 2 * padding_y)})
    layout = layout_labels(image_width, image_height, [r["rect"] for r in rows],
                           [r["size"] for r in rows], person_padding=round(7 * unit), label_gap=round(6 * unit))
    footer_height = round(100 * unit)
    canvas = Image.new("RGB", (layout.width, layout.height + footer_height), "#112E29")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    line_width = max(1, round(2 * unit))
    # Lines terminate at the corresponding box, never at a person's face.
    for placement in layout.placements:
        if len(placement.connector) > 1:
            color = rows[placement.target_index]["color"]
            draw.line(placement.connector, fill="#173A31", width=line_width + 2, joint="curve")
            draw.line(placement.connector, fill=color, width=line_width, joint="curve")
    for row in rows:
        draw.rectangle(row["rect"], outline="#173A31", width=line_width + 2)
        draw.rectangle(row["rect"], outline=row["color"], width=line_width)
    for placement in layout.placements:
        row = rows[placement.target_index]
        rect = placement.rect
        draw.rounded_rectangle(rect, radius=round(5 * unit), fill="#123A30", outline=row["color"], width=1)
        draw.text((rect[0] + padding_x, rect[1] + padding_y), row["label"], font=font, fill="#FFFFFF", anchor="lt")
        if placement.connector:
            x, y = placement.connector[-1]
            radius = max(2, round(3 * unit))
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=row["color"])
    top = layout.height
    inset = round(25 * unit)
    draw.line((0, top, canvas.width, top), fill="#35574B", width=1)
    count_text = str(observation["people"])
    count_x, count_y = inset, top + round(20 * unit)
    draw.text((count_x, count_y), count_text, font=count_font, fill="#A7EBCB", anchor="lt")
    next_x = count_x + round(draw.textlength(count_text, font=count_font)) + round(11 * unit)
    draw.text((next_x, count_y + round(5 * unit)), "people in this frame", font=caption_font, fill="#F1F7F4", anchor="lt")
    detail = "Detection numbers apply only to this frame" if annotated else "Source pixels removed · Detection numbers apply only to this frame"
    draw.text((inset, top + round(63 * unit)), detail, font=small_font, fill="#A8C0B5", anchor="lt")
    timestamp = f"{observation['time']:.2f} s"
    badge_width = round(draw.textlength(timestamp, font=font)) + round(30 * unit)
    badge = (canvas.width - inset - badge_width, top + round(22 * unit), canvas.width - inset, top + round(60 * unit))
    draw.rounded_rectangle(badge, radius=round(8 * unit), fill="#234B3F", outline="#406858", width=1)
    draw.text((badge[0] + round(15 * unit), badge[1] + round(10 * unit)), timestamp, font=font, fill="#D1E8DD", anchor="lt")
    return canvas
