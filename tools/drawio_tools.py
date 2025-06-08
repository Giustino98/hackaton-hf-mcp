import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    # Consider raising an error or logging if the API key is critical for module loading
    print("Attenzione: GEMINI_API_KEY non trovato nelle variabili d'ambiente.")
    # raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Utilizziamo lo stesso modello specificato in object_detection_tools.py per coerenza,
# o un modello potente per la generazione come "gemini-1.5-pro-latest".
# Se "gemini-2.5-pro-preview-06-05" è disponibile e preferito:
MODEL_NAME = "gemini-2.5-pro-preview-06-05"
# Altrimenti, un'opzione robusta:
# MODEL_NAME = "gemini-1.5-pro-latest"

try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Errore durante l'inizializzazione del client GenAI: {e}")
    client = None # o gestire l'errore come appropriato

DRAWIO_SYSTEM_INSTRUCTIONS = """
You are an expert Draw.io diagram generator.
Your task is to create a Draw.io XML representation of an image, based on the original image and a list of identified objects within it.
The output must be a single, valid Draw.io XML string, without any surrounding text or markdown.
Do not include any other text, explanations, or markdown code fencing (e.g., ```xml ... ```) around the XML.
The diagram should represent the spatial arrangement and composition of the original image. The Draw.io file and the image assets are both located in the 'output_llm' subdirectory.
Therefore, image paths within the Draw.io XML must be relative to the location of the Draw.io file itself. Since both are in 'output_llm', the path should be just the filename, e.g., 'object_name.png'.
Use the object name (e.g., 'object_name' extracted from '../output_llm/object_name.png') as the `value` (label) for the `mxCell` if a label is desired, or leave `value` empty for image-only objects.

A basic Draw.io XML structure looks like this (ensure `compressed="false"`):
<mxfile compressed="false" host="GeminiAgent" version="1.0" type="device">
  <diagram id="diagram-1" name="Page-1">
    <mxGraphModel dx="1000" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <!-- Example of a generic shape object -->
        <mxCell id="obj_rect_id_1" value="LabelForRect" style="shape=rectangle;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="200" y="150" width="120" height="60" as="geometry" />
        </mxCell>
        <!-- Example of an embedded image object:
             'value' can be the object name (e.g., 'example_object') or empty.
             'image' attribute in style must point to the relative path of the image asset.
             The path should be like 'example_object.png'.
             'shape=image' is crucial. 'imageAspect=0' allows stretching if needed, 'imageAspect=1' preserves aspect ratio.
             'aspect=fixed' can be used if the geometry itself should maintain its aspect ratio when resized manually. -->
        <mxCell id="obj_img_id_1" value="example_object" style="shape=image;html=1;imageAspect=0;image=example_object.png;" vertex="1" parent="1">
          <mxGeometry x="400" y="150" width="100" height="80" as="geometry" />
        </mxCell>
        <!-- Add more mxCell elements for other objects and their relationships. Ensure unique 'id' for each mxCell. -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
Adapt this structure. Ensure all `mxCell` elements have unique `id` attributes. Ensure image paths are direct filenames like 'asset.png', as they are in the same directory as the .drawio file.
The `x`, `y`, `width`, `height` attributes in `mxGeometry` should reflect the object's approximate position and size in the original image.
When using an image asset (e.g., from '../output_llm/object.png'), its `mxGeometry` should ideally correspond to the detected object's bounding box in the original image, scaled appropriately for the diagram.
"""

@tool("drawio_generator_tool")
def generate_drawio_from_image_and_objects(original_image_path: str, object_names: list[str]) -> str:
    """
    Genera una rappresentazione XML Draw.io di un'immagine,
    utilizzando l'immagine originale e una lista di nomi di oggetti identificati.

    Args:
        original_image_path (str): Il percorso del file dell'immagine originale.
        object_names (list[str]): Una lista di nomi di file (con estensione, es. "gatto.png", "sedia.jpg")
                                   per le immagini degli oggetti precedentemente estratti e salvati nella cartella "output_llm/".
                                   Questi verranno usati come asset nel diagramma.

    Returns:
        str: Una stringa XML Draw.io che rappresenta l'immagine, o un messaggio di errore.
    """
    if not GOOGLE_API_KEY or not client:
        return "Errore: GEMINI_API_KEY non configurato o client non inizializzato."

    try:
        with open(original_image_path, "rb") as f:
            img_bytes = f.read()
        original_image = Image.open(BytesIO(img_bytes))
        original_image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        object_image_folder = "output_llm"  # Cartella dove sono salvate le immagini degli oggetti

        prompt_parts = [
            "Generate a Draw.io XML diagram for the provided original image.",
            "The diagram should represent the overall scene, focusing on spatial relationships and composition."
        ]

        if object_names: # e.g., ["cat.png", "dog.png"]
            # Create a string like "'cat.png', 'dog.png'" for the prompt
            object_filenames_str = ", ".join([f"'{name}'" for name in object_names])
            prompt_parts.append(
                f"Incorporate the following pre-detected object images as assets in your diagram: {object_filenames_str}."
            )
            prompt_parts.append(
                f"These image assets (e.g., 'cat.png') are located in the '{object_image_folder}' directory. "
                f"The Draw.io XML file itself will also be saved in this same '{object_image_folder}' directory."
            )
            prompt_parts.append(
                f"Therefore, when referencing an asset like 'cat.png' in the Draw.io XML, the 'image' attribute "
                f"within the 'style' string of an 'mxCell' element must be just the filename. "
                f"For example: style='shape=image;html=1;imageAspect=0;image=cat.png;'."
            )
            prompt_parts.append(
                f"For the 'value' attribute (label) of these image mxCell elements, you can use the base name of the object "
                f"(e.g., for 'cat.png', use 'cat' as the label), or leave it empty if a label is not visually appropriate."
            )

        prompt_parts.append("Arrange all elements to reflect their spatial relationships and the overall composition of the original scene.")
        prompt_parts.append("The diagram should visually correspond to the input image.")
        user_prompt = " ".join(prompt_parts)

        response = client.models.generate_content(
            contents=[user_prompt, original_image],
            model=MODEL_NAME,
                config = types.GenerateContentConfig( # Corretto da 'config' a 'generation_config'
                system_instruction=DRAWIO_SYSTEM_INSTRUCTIONS, # system_instruction non è un parametro di GenerationConfig
                temperature=0,
                safety_settings=[ # Impostazioni di sicurezza simili a quelle di object_detection_tools.py
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            ]
            )
        )

        xml_output = response.text.strip()
        if xml_output.startswith("```xml"): xml_output = xml_output[len("```xml"):]
        if xml_output.endswith("```"): xml_output = xml_output[:-len("```")]
        return xml_output.strip()

    except FileNotFoundError:
        return f"Errore: File immagine originale non trovato a {original_image_path}."
    except Exception as e:
        print(f"Errore dettagliato in generate_drawio_from_image_and_objects: {e}")
        return f"Errore durante la generazione dell'XML Draw.io: {str(e)}"

@tool("drawio_saver_tool")
def save_drawio_xml(xml_content: str, filename_prefix: str, output_directory: str = "output_llm") -> str:
    """
    Salva una stringa XML di Draw.io in un file .drawio.

    Args:
        xml_content (str): La stringa XML del diagramma Draw.io.
        filename_prefix (str): Il prefisso per il nome del file. Il file verrà salvato come '{filename_prefix}.drawio'.
        output_directory (str): La directory dove salvare il file. Default 'output_llm'.

    Returns:
        str: Il percorso del file salvato o un messaggio di errore.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Assicurati che il nome del file finisca con .drawio
        if not filename_prefix.endswith(".drawio"):
            filename = f"{filename_prefix}.drawio"
        else:
            filename = filename_prefix

        file_path = os.path.join(output_directory, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        return f"File Draw.io salvato con successo in: {os.path.abspath(file_path)}"
    except Exception as e:
        return f"Errore durante il salvataggio del file Draw.io: {str(e)}"