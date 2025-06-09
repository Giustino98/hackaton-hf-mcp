import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool

# utils.utils.plot_bounding_boxes non è necessario per il tool in sé, ma per la visualizzazione
from utils.utils import plot_bounding_boxes, save_cropped_images

GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

model_name = "gemini-2.5-pro-preview-06-05" # @param ["gemini-1.5-flash-latest","gemini-2.0-flash-lite","gemini-2.0-flash","gemini-2.5-flash-preview-05-20","gemini-2.5-pro-preview-06-05"] {"allow-input":true}
# System instructions per guidare il modello a restituire i bounding box in formato JSON.
bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If two object are the same, understand if they have different labels or not, and if they do, return them as two different objects, othewrise return just one object.
    The image you are analyzing is a 2D diagram image, you should return the boung boxes of the singular pics of the architecture,
    not the composition of multiple pics or the whole image.
    The label you assign should be as more specific as possible to allow the user to understand what the object is. If there is any space in the label, insert underscore between the words.
    If some pics in the images contain labels, try to exclude them from the bounding boxes (unless they overlap with the object you are bounding).
      """

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

@tool("object_detection_tool", parse_docstring=True)
def detect_objects_in_image(img_path: str) -> str:
    """
    Detects objects in an image and returns their 2D bounding boxes along with labels.

    Args:
        img_path (str): The path to the image file.

    Returns:
        str: A JSON string containing the detected objects' bounding boxes and labels,
             or an error message if detection fails.
    """

    user_prompt: str = "Detect the 2d bounding boxes of the objects in the image (with “label” as object description)."
    if not GOOGLE_API_KEY:
        return "Error: GEMINI_API_KEY not configured."
    try:
        # Load and resize image
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        im = Image.open(BytesIO(img_bytes))
        im.thumbnail([1024,1024], Image.Resampling.LANCZOS)

        # Run model to find bounding boxes
        response = client.models.generate_content(
            model=model_name,
            contents=[user_prompt, im],
            config = types.GenerateContentConfig( # Corretto da 'config' a 'generation_config'
                system_instruction=bounding_box_system_instructions, # system_instruction non è un parametro di GenerationConfig
                temperature=0,
                safety_settings=safety_settings,
            )
        )
        
        save_cropped_images(im, response.text, output_folder="output_llm")
        plot_bounding_boxes(im, response.text)

        return response.text
    except FileNotFoundError:
        return f"Error: Image file not found at {img_path}."
    except Exception as e:
        return f"Error detecting objects: {str(e)}"
