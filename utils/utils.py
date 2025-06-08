import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
import os
from PIL import ImageColor

# Costante per il fattore di normalizzazione usato nelle coordinate
NORMALIZATION_DIVISOR = 1000

# @title Parsing JSON output
def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

# @title Plotting Util

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["box_2d"][0] / NORMALIZATION_DIVISOR * height)
      abs_x1 = int(bounding_box["box_2d"][1] / NORMALIZATION_DIVISOR * width)
      abs_y2 = int(bounding_box["box_2d"][2] / NORMALIZATION_DIVISOR * height)
      abs_x2 = int(bounding_box["box_2d"][3] / NORMALIZATION_DIVISOR * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)

    # Display the image
    img.show()

def save_cropped_images(
    im: Image.Image, bounding_boxes_json_str: str, output_folder: str = "output_llm"
) -> list[str]:
    """
    Ritaglia oggetti da un'immagine in base alle bounding box e li salva in una cartella specificata.

    Args:
        im: L'oggetto PIL.Image.
        bounding_boxes_json_str: Una stringa JSON contenente le bounding box.
                                 Ogni box dovrebbe avere "label" e "box_2d"
                                 (coordinate normalizzate [y1, x1, y2, x2] su base NORMALIZATION_DIVISOR).
        output_folder: La cartella dove verranno salvate le immagini ritagliate. Default "files".

    Returns:
        list[str]: Una lista dei percorsi ai file delle immagini ritagliate salvate con successo.
    """
    saved_file_paths = []
    os.makedirs(output_folder, exist_ok=True)
    width, height = im.size

    # Parsing della stringa JSON
    parsed_json_str = parse_json(bounding_boxes_json_str)
    try:
        bounding_boxes_list = json.loads(parsed_json_str)
    except json.JSONDecodeError as e:
        print(f"Errore nel decodificare JSON: {e}")
        return saved_file_paths # Ritorna lista vuota in caso di errore JSON iniziale

    filename_counts = {}  # Per gestire etichette duplicate

    for i, bounding_box in enumerate(bounding_boxes_list):
        if "box_2d" not in bounding_box:
            print(f"Bounding box {i} saltata: chiave 'box_2d' mancante.")
            continue
        if len(bounding_box["box_2d"]) != 4:
            print(f"Bounding box {i} saltata: 'box_2d' non ha 4 coordinate.")
            continue

        # Converte coordinate normalizzate in coordinate assolute
        # box_2d è [y1, x1, y2, x2]
        abs_y1 = int(bounding_box["box_2d"][0] / NORMALIZATION_DIVISOR * height)
        abs_x1 = int(bounding_box["box_2d"][1] / NORMALIZATION_DIVISOR * width)
        abs_y2 = int(bounding_box["box_2d"][2] / NORMALIZATION_DIVISOR * height)
        abs_x2 = int(bounding_box["box_2d"][3] / NORMALIZATION_DIVISOR * width)

        # Assicura che abs_x1 sia sinistra, abs_x2 destra, abs_y1 alto, abs_y2 basso
        # per la funzione crop di PIL che richiede (left, upper, right, lower)
        crop_left = min(abs_x1, abs_x2)
        crop_upper = min(abs_y1, abs_y2)
        crop_right = max(abs_x1, abs_x2)
        crop_lower = max(abs_y1, abs_y2)

        if crop_left >= crop_right or crop_upper >= crop_lower:
            label_for_log = bounding_box.get('label', f'indice {i}')
            print(f"Bounding box per '{label_for_log}' saltata: area nulla ({crop_left},{crop_upper},{crop_right},{crop_lower})")
            continue

        cropped_image = im.crop((crop_left, crop_upper, crop_right, crop_lower))

        label = bounding_box.get("label", f"unlabeled_crop_{i}")
        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        if not safe_label:
            safe_label = f"unlabeled_crop_{i}"

        count = filename_counts.get(safe_label, 0)
        filename_counts[safe_label] = count + 1
        output_filename = f"{safe_label}_{count}.png" if count > 0 else f"{safe_label}.png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            cropped_image.save(output_path)
            saved_file_paths.append(output_path)
        except Exception as e:
            print(f"Errore nel salvare l'immagine {output_path}: {e}")

    return saved_file_paths