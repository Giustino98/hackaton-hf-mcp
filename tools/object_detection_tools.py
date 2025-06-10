import json
import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool
import os
from langchain_core.tools import tool
import re # Aggiunto import per regex
from typing import Any, Dict, List
from utils.utils import parse_json, plot_bounding_boxes, save_cropped_images
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-06-05")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "128"))

client = genai.Client(api_key=GOOGLE_API_KEY)

# Langfuse initialization
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000") # Default to local if not set

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        print(f"Langfuse tracing enabled for {__name__}")
    except Exception as e:
        print(f"Failed to initialize Langfuse for {__name__}: {e}. Tracing will be disabled.")
# System instructions per guidare il modello a restituire i bounding box in formato JSON.
bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    The image you are analyzing is a 2D diagram image, you must return the bounding boxes of the singular pics of the architecture,
    not the composition of multiple pics or the whole image. You must detect the objects in the image as more as precise as possible, excluding background, arrows and text from the boxes.
    In case you are not sure with coordinates, create looser bounding boxes but do not crop the objects.
    The label you assign should be as more specific as possible to allow the user to understand what the object is. If there is any space in the label, insert underscore between the words.
"""

adjust_bounding_box_system_instructions = """
You are an expert in refining 2D bounding boxes for objects in an image.
You will be given an image, a set of existing bounding boxes (in JSON format: {"label": "object_name", "box_2d": [y1, x1, y2, x2]}), and a user prompt describing necessary corrections.
Your task is to return a new JSON list of bounding boxes that incorporates the user's corrections.
The "box_2d" coordinates are normalized to 1000.
Focus on applying the corrections accurately. You can adjust existing boxes, add new ones if implied by the corrections, or remove boxes if implied.
If a label has spaces, insert underscores instead.
Return ONLY the corrected JSON list of bounding boxes, and no other text before or after the JSON block.
"""

verification_system_instructions = """
You are an expert image analysis assistant. Your task is to evaluate the provided 2D bounding boxes against the given image.
The bounding boxes are provided as a JSON string, where each box has a "label" and "box_2d" coordinates.
The "box_2d" coordinates are in the format [y1, x1, y2, x2] and are normalized to 1000 (e.g., a y1 of 500 means the top edge is at the vertical midpoint of the image).
The bounding boxes extracted should be related to the single components and not composition of objects, labels or structural elements.

Carefully examine the image and the bounding boxes.
Determine if the bounding boxes are:
1.  **Accurate**: Do they tightly enclose the intended objects without including too much background or cutting off parts of the objects?
2.  **Complete**: Are all relevant and distinct objects in the image correctly identified and bounded? Are there any missing objects that should be captured? Are there redundant or overlapping boxes for the same object instance?
3.  **Correctly Labeled**: Do the labels accurately describe the objects?

In case the majority of bounding boxes are satisfactory, answer that the result is pretty quite good and no suggestions are needed.

Respond ONLY in JSON format with the following structure, and no other text before or after the JSON block:
{
  "is_satisfactory": boolean,
  "suggestions": [],
  "reasoning": "A brief explanation of your evaluation."
}

Example for unsatisfactory boxes:
{
  "is_satisfactory": false,
  "suggestions": [
    "The bounding box for 'widget_A' is too large and includes excessive background on the right.",
    "A 'connector_cable' object appears to be present between 'widget_A' and 'widget_B' but is not bounded.",
    "The label 'button_3' seems to be a duplicate or too similar to 'button_2'; consider merging or clarifying."
  ],
  "reasoning": "Several boxes are inaccurate, one object is missing, and there might be a redundant label."
}

Example for satisfactory boxes:
{
  "is_satisfactory": true,
  "suggestions": [],
  "reasoning": "All objects are accurately bounded, distinctly captured, and correctly labeled according to the image content."
}
"""



safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

@tool("object_detection_tool", parse_docstring=True)
@observe(as_type="generation")
def detect_objects_in_image(img_path: str, user_prompt: str = "Detect the 2d bounding boxes of the objects in the image (with “label” as object description).") -> str:
    """
    Detects objects in an image and returns their 2D bounding boxes along with labels.
    It saves the cropped images of detected objects that will be then referenced to generate a drawio diagram.

    Args:
        img_path (str): The path to the image file.

    Returns:
        str: A JSON string containing the detected objects' bounding boxes and labels,
             or an error message if detection fails.
    """

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
            model=GEMINI_MODEL_NAME,
            contents=[user_prompt, im],
            config = types.GenerateContentConfig( # Corretto da 'config' a 'generation_config'
                system_instruction=bounding_box_system_instructions, # system_instruction non è un parametro di GenerationConfig
                temperature=0,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
            )
        )

        langfuse_context.update_current_observation(
            input=[user_prompt, im],
            model=model_name,
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count
            }
        )
        
        save_cropped_images(im, response.text, output_folder="output_llm")
        plot_bounding_boxes(im, response.text)

        return response.text
    except FileNotFoundError:
        return f"Error: Image file not found at {img_path}."
    except Exception as e:
        return f"Error detecting objects: {str(e)}"

@tool("adjust_bounding_boxes_tool", parse_docstring=True)
@observe(as_type="generation")
def adjust_bounding_boxes_in_image(img_path: str, existing_bounding_boxes_json_str: str, user_prompt: str) -> str:
    """
    Adjusts existing bounding boxes for objects in an image based on user feedback.
    It takes an image, a JSON string of current bounding boxes, and a user prompt detailing corrections.
    It returns a new JSON string with the adjusted bounding boxes and saves the new cropped images.

    Args:
        img_path (str): The path to the image file.
        existing_bounding_boxes_json_str (str): A JSON string of the current bounding boxes.
                                                This string might contain markdown ```json ... ```, which will be cleaned.
        user_prompt (str): User instructions on how to correct the bounding boxes
                           (e.g., "Make the box for 'obj1' smaller and move 'obj2' to the right.").

    Returns:
        str: A JSON string containing the adjusted objects' bounding boxes and labels,
             or an error message if adjustment fails.
    """
    if not GOOGLE_API_KEY:
        return "Error: GEMINI_API_KEY not configured."

    try:
        # Load and resize image
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        im = Image.open(BytesIO(img_bytes))
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        # Pre-process the input JSON string for robustness
        processed_input_json_str = existing_bounding_boxes_json_str.strip()
        if processed_input_json_str.lower().startswith("json\n"):
            # Handles cases like "json\n[...]"
            processed_input_json_str = processed_input_json_str[5:].strip()
        elif processed_input_json_str.lower().startswith("```json"):
            # Handles cases like "```json\n[...]\n```" more explicitly before parse_json
            # parse_json should ideally handle this, but this adds robustness
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", processed_input_json_str, re.DOTALL | re.IGNORECASE)
            if match:
                processed_input_json_str = match.group(1).strip()
            # If no match, parse_json will try to handle it or it might be clean already

        # parse_json (from utils) should handle standard markdown block cleaning
        # It's called here in case the pre-processing didn't fully clean it or for other markdown cases.
        cleaned_existing_boxes_str_for_load = parse_json(processed_input_json_str)

        # Prepare prompt for the model
        prompt_for_adjustment = [
            user_prompt,
            im,
            f"Existing bounding boxes to adjust:\n{cleaned_existing_boxes_str_for_load}"
        ]

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt_for_adjustment,
            config=types.GenerateContentConfig(
                system_instruction=adjust_bounding_box_system_instructions,
                temperature=0,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
            )
        )

        langfuse_context.update_current_observation(
            input=prompt_for_adjustment,
            model=model_name, # Or your specific adjustment model if different
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count
            }
        )

        adjusted_boxes_str_from_model = parse_json(response.text) # Clean markdown from model response
        save_cropped_images(im, adjusted_boxes_str_from_model, output_folder="output_llm") # Save new crops
        plot_bounding_boxes(im, adjusted_boxes_str_from_model) # Plot new boxes

        return adjusted_boxes_str_from_model
    except FileNotFoundError:
        return f"Error: Image file not found at {img_path}."
    except json.JSONDecodeError as e:
        # Provide more context for debugging JSON errors related to existing_bounding_boxes
        # The error likely occurred when trying to process/validate existing_bounding_boxes_json_str
        # or its derivatives before sending to the model, or if parse_json itself tries to json.loads.
        return (f"Error: Invalid JSON format for existing bounding boxes. "
                f"Attempted to process: '{processed_input_json_str if 'processed_input_json_str' in locals() else existing_bounding_boxes_json_str}'. "
                f"Output of parse_json: '{cleaned_existing_boxes_str_for_load if 'cleaned_existing_boxes_str_for_load' in locals() else 'N/A'}'. "
                f"JSONDecodeError: {str(e)}")
    except Exception as e:
        return f"Error adjusting bounding boxes: {str(e)}. Input existing_bounding_boxes_json_str was: '{existing_bounding_boxes_json_str}'"

@tool("delete_cropped_images_tool", parse_docstring=True)
def delete_cropped_images(image_paths: List[str], base_folder: str = "output_llm") -> str:
    """
    Deletes a list of specified cropped image files from the 'output_llm' folder.

    Args:
        image_paths (List[str]): A list of filenames (e.g., ['cat.png', 'dog.png'])
                                 of the cropped images to be deleted.
        base_folder (str): The base directory where the images are stored.
                           Defaults to "output_llm".

    Returns:
        str: A message indicating the outcome of the deletion process,
             including counts of successfully deleted files and any errors.
    """
    deleted_count = 0
    errors = []
    
    if not image_paths:
        return "Nessun percorso immagine fornito per l'eliminazione."

    for image_name in image_paths:
        # Assicurati che il percorso sia relativo alla base_folder se non è già assoluto
        if not os.path.isabs(image_name):
            file_path = os.path.join(base_folder, image_name)
        else:
            file_path = image_name
            
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"Immagine eliminata con successo: {file_path}")
            else:
                errors.append(f"File non trovato: {file_path}")
                print(f"Attenzione: File non trovato, impossibile eliminare: {file_path}")
        except OSError as e:
            errors.append(f"Errore durante l'eliminazione di {file_path}: {e}")
            print(f"Errore durante l'eliminazione di {file_path}: {e}")
        except Exception as e:
            errors.append(f"Errore imprevisto durante l'eliminazione di {file_path}: {e}")
            print(f"Errore imprevisto durante l'eliminazione di {file_path}: {e}")

    result_message = f"Processo di eliminazione completato. Immagini eliminate: {deleted_count}."
    if errors:
        result_message += " Errori riscontrati:\n" + "\n".join(errors)
    
    return result_message


@tool("verify_bounding_boxes_tool", parse_docstring=True)
@observe(as_type="generation")
def verify_bounding_boxes(
    img_path: str,
    bounding_boxes_json_str: str,
    user_query: str = "Evaluate the accuracy and completeness of these bounding boxes for the given image. Provide specific suggestions for improvement if they are not satisfactory."
) -> Dict[str, Any]:
    """
    Verifies if the provided bounding boxes are satisfactory for the given image using a multimodal model.
    It compares the image with the bounding boxes and returns an evaluation.

    Args:
        img_path (str): The path to the image file.
        bounding_boxes_json_str (str): A JSON string of the detected bounding boxes.
                                      This string might contain markdown ```json ... ```, which will be cleaned.
        user_query (str): Optional query to guide the verification process.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        - is_satisfactory (bool): True if the bounding boxes are deemed satisfactory.
                        - suggestions (List[str]): A list of suggestions for improvement if not satisfactory.
                        - reasoning (str): The model's reasoning for its evaluation.
                        - raw_model_response (str): The raw text response from the model.
                        Returns an error structure if the process fails.
    """
    if not GOOGLE_API_KEY or not client:
        return {
            "is_satisfactory": False,
            "suggestions": ["GEMINI_API_KEY not configured or client not initialized."],
            "reasoning": "Configuration error.",
            "raw_model_response": "GEMINI_API_KEY not configured or client not initialized."
        }

    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        im = Image.open(BytesIO(img_bytes))
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        cleaned_bounding_boxes_json_str = parse_json(bounding_boxes_json_str)
        try:
            json.loads(cleaned_bounding_boxes_json_str) # Validate JSON before sending
        except json.JSONDecodeError:
            return {
                "is_satisfactory": False,
                "suggestions": ["Invalid bounding_boxes_json_str: Not valid JSON after cleaning markdown."],
                "reasoning": "Input data error.",
                "raw_model_response": bounding_boxes_json_str 
            }

        prompt_parts = [
            im,
            f"Here are the bounding boxes to verify (normalized [y1,x1,y2,x2] format, 0-1000 scale):\n{cleaned_bounding_boxes_json_str}",
            f"\nUser query/instruction for verification: {user_query}"
        ]

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt_parts,
            config = types.GenerateContentConfig( # Corretto da 'config' a 'generation_config'
                system_instruction=verification_system_instructions, # system_instruction non è un parametro di GenerationConfig
                temperature=0,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
            )
        )

        langfuse_context.update_current_observation(
            input=prompt_parts,
            model=model_name,
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count
            }
        )

        raw_model_text = response.text
        try:
            parsed_model_response_text = parse_json(raw_model_text) # Clean potential markdown from model
            model_response_json = json.loads(parsed_model_response_text)
            
            is_satisfactory = model_response_json.get("is_satisfactory", False)
            suggestions = model_response_json.get("suggestions", [])
            reasoning = model_response_json.get("reasoning", "")

            if not isinstance(suggestions, list): # Ensure suggestions is a list
                suggestions = [str(suggestions)] if suggestions else []
            if is_satisfactory and suggestions: # Enforce LLM instruction
                 print(f"Warning: Model returned is_satisfactory=True but also suggestions: {suggestions}. Clearing suggestions.")
                 suggestions = []

            return {
                "is_satisfactory": is_satisfactory,
                "suggestions": suggestions,
                "reasoning": reasoning,
                "raw_model_response": raw_model_text,
            }
        except json.JSONDecodeError as e:
            return {
                "is_satisfactory": False,
                "suggestions": [f"Model did not return valid JSON as expected. Parse error: {e}. Raw response: {raw_model_text}"],
                "reasoning": "Failed to parse model's JSON output.",
                "raw_model_response": raw_model_text,
            }
    except FileNotFoundError:
        return {"is_satisfactory": False, "suggestions": [f"Image file not found at {img_path}."], "reasoning": "File error.", "raw_model_response": f"Image file not found at {img_path}."}
    except Exception as e:
        print(f"Error during bounding box verification: {str(e)}")
        return {
            "is_satisfactory": False,
            "suggestions": [f"An error occurred during verification: {str(e)}"],
            "reasoning": "Execution error.",
            "raw_model_response": str(e),
       }


@tool("select_latest_image_versions_tool", parse_docstring=True)
def select_latest_image_versions(directory: str = "output_llm") -> List[str]:
    """
    Scans a directory for image files (png, jpg, jpeg, gif, bmp) and returns a list
    containing the latest version of each image. For example, if 'file.png', 'file_01.png',
    and 'file_2.png' exist, only 'file_2.png' (or the one with the highest number)
    will be included in the returned list. Files without numbered versions are included as is.

    Args:
        directory (str): The directory to scan. Defaults to "output_llm".

    Returns:
        List[str]: A list of filenames representing the latest version of each image.
                   Returns an empty list if the directory doesn't exist or no images are found.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return []

    image_files = {}
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            base, ext = os.path.splitext(filename)
            # Regex per trovare file con _numero o -numero prima dell'estensione
            # es. file_01, file-1, file_2
            match = re.match(r"^(.*?)[\s_-]*(\d+)$", base)
            if match:
                name_part = match.group(1)
                version_number = int(match.group(2))
            else:
                name_part = base
                version_number = -1 # Considera i file senza numero come versione -1 (o 0 se preferisci)

            # Usa una tupla (name_part, ext) come chiave per raggruppare le versioni
            file_key = (name_part.lower(), ext.lower())

            if file_key not in image_files or version_number > image_files[file_key]['version']:
                image_files[file_key] = {'filename': filename, 'version': version_number, 'original_base': name_part, 'original_ext': ext}

    # Estrai i nomi dei file finali
    latest_versions = [data['filename'] for data in image_files.values()]
    
    print(f"Selected latest image versions from '{directory}': {latest_versions}")
    return latest_versions