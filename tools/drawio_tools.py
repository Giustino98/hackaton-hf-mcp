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

import base64
import mimetypes
import os
import re
from xml.etree import ElementTree as ET
from typing import Optional

def convert_image_to_base64(image_path: str) -> Optional[str]:
    """
    Converte un'immagine in stringa base64 nel formato Draw.io
    
    Args:
        image_path: Percorso del file immagine
        
    Returns:
        Stringa base64 nel formato Draw.io (data:image/type,base64_data) o None se errore
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found")
        return None
        
    try:
        # Determina il MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/png'  # default fallback
        
        # Leggi e converti in base64
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode('utf-8')
            
        # Formato Draw.io: data:image/type,base64_data (SENZA ;base64)
        return f"data:{mime_type},{base64_str}"
        
    except Exception as e:
        print(f"Error converting {image_path} to base64: {e}")
        return None

def replace_image_references_in_drawio_xml(xml_content: str, base_folder: str = "output_llm") -> str:
    """
    Sostituisce tutti i riferimenti alle immagini nell'XML Draw.io con versioni base64
    
    Args:
        xml_content: Contenuto XML Draw.io come stringa
        base_folder: Cartella base dove cercare le immagini
        
    Returns:
        XML modificato con immagini base64 embedded
    """
    try:
        # Pattern per trovare riferimenti alle immagini negli attributi style
        # Cerca pattern come: image=filename.png o image='filename.png' o image="filename.png"
        image_patterns = [
            r'image=([\'"]?)([^\'";,\s]+\.(png|jpg|jpeg|gif|bmp|svg))\1',  # image=file.png, image='file.png', image="file.png"
            r'image=([\'"]?)(file://\.?/?([^\'";,\s]+\.(png|jpg|jpeg|gif|bmp|svg)))\1',  # image=file://./file.png
        ]
        
        modified_xml = xml_content
        processed_files = set()  # Per evitare conversioni duplicate
        
        for pattern in image_patterns:
            matches = re.finditer(pattern, modified_xml, re.IGNORECASE)
            
            for match in matches:
                full_match = match.group(0)
                quote_char = match.group(1) if match.group(1) else ''
                
                # Estrai il nome del file
                if 'file://' in full_match:
                    # Per pattern file://./filename.png
                    filename = match.group(3) if len(match.groups()) >= 3 else match.group(2)
                else:
                    # Per pattern semplici
                    filename = match.group(2)
                
                # Rimuovi eventuali prefissi di path
                filename = os.path.basename(filename)
                
                if filename in processed_files:
                    continue
                    
                processed_files.add(filename)
                image_path = os.path.join(base_folder, filename)
                
                # Converti in base64
                base64_data = convert_image_to_base64(image_path)
                
                if base64_data:
                    # Sostituisci tutti i riferimenti a questo file
                    old_patterns = [
                        f'image={quote_char}{filename}{quote_char}',
                        f'image={quote_char}file://\./{filename}{quote_char}',
                        f'image={quote_char}file://{filename}{quote_char}',
                        f'image={filename}',  # senza quote
                    ]
                    
                    new_value = f'image={quote_char}{base64_data}{quote_char}' if quote_char else f'image={base64_data}'
                    
                    for old_pattern in old_patterns:
                        modified_xml = modified_xml.replace(old_pattern, new_value)
                    
                    print(f"Replaced image reference: {filename} -> base64 ({len(base64_data)} chars)")
                else:
                    print(f"Failed to convert image: {filename}")
        
        return modified_xml
        
    except Exception as e:
        print(f"Error processing XML: {e}")
        return xml_content  # Ritorna l'originale in caso di errore

def replace_image_references_xml_parser(xml_content: str, base_folder: str = "output_llm") -> str:
    """
    Versione alternativa che usa XML parser per maggiore precisione
    Sostituisce i riferimenti alle immagini negli attributi style dei mxCell
    """
    try:
        # Parse dell'XML
        root = ET.fromstring(xml_content)
        
        # Trova tutti gli elementi mxCell con attributo style contenente image=
        for cell in root.iter('mxCell'):
            style = cell.get('style', '')
            if 'image=' in style:
                # Estrai il valore dell'immagine dallo style
                style_parts = style.split(';')
                new_style_parts = []
                
                for part in style_parts:
                    if part.startswith('image='):
                        # Estrai il nome del file
                        image_ref = part[6:]  # Rimuovi 'image='
                        
                        # Rimuovi eventuali quote
                        if image_ref.startswith('"') and image_ref.endswith('"'):
                            image_ref = image_ref[1:-1]
                        elif image_ref.startswith("'") and image_ref.endswith("'"):
                            image_ref = image_ref[1:-1]
                        
                        # Gestisci file:// prefix
                        if image_ref.startswith('file://'):
                            image_ref = image_ref.replace('file://', '').lstrip('./')
                        
                        filename = os.path.basename(image_ref)
                        image_path = os.path.join(base_folder, filename)
                        
                        # Converti in base64
                        base64_data = convert_image_to_base64(image_path)
                        
                        if base64_data:
                            new_style_parts.append(f'image={base64_data}')
                            print(f"XML Parser: Replaced {filename} with base64 data")
                        else:
                            new_style_parts.append(part)  # Mantieni originale se conversione fallisce
                    else:
                        new_style_parts.append(part)
                
                # Ricostruisci lo style
                cell.set('style', ';'.join(new_style_parts))
        
        # Converti back in stringa
        return ET.tostring(root, encoding='unicode')
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        # Fallback al metodo regex
        return replace_image_references_in_drawio_xml(xml_content, base_folder)
    except Exception as e:
        print(f"Error in XML parser method: {e}")
        return xml_content

@tool("generate_drawio_from_image_and_objects_tool", parse_docstring=True) # Uncomment if you plan to use it directly as a langchain tool
def generate_drawio_from_image_and_objects(original_image_path: str, object_names: list[str]) -> str:
    """
    Generates a Draw.io XML diagram from an original image and a list of detected object names.

    The function first instructs a generative model to create a Draw.io XML representation
    of the scene in the original image. It then incorporates references to cropped images
    of specified objects (expected to be in the 'output_llm' folder).
    Finally, it post-processes this XML to replace all local image file references
    with their base64 encoded data, making the Draw.io diagram self-contained.

    Args:
        original_image_path (str): The file path to the original image to be diagrammed.
        object_names (list[str]): A list of object names (e.g., ['cat.png', 'dog.png']) that have been previously detected and saved as image files in the 'output_llm' folder. These will be embedded into the diagram.

    Returns:
        bool: True if the Draw.io XML was successfully generated and saved, or an error message if something went wrong.
    """
    if not GOOGLE_API_KEY or not client:
        return "Errore: GEMINI_API_KEY non configurato o client non inizializzato."

    try:
        with open(original_image_path, "rb") as f:
            img_bytes = f.read()
        original_image = Image.open(BytesIO(img_bytes))
        original_image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        object_image_folder = "output_llm"

        prompt_parts = [
            "Generate a Draw.io XML diagram for the provided original image.",
            "The diagram should represent the overall scene, focusing on spatial relationships and composition."
        ]

        if object_names:
            object_filenames_str = ", ".join([f"'{name}'" for name in object_names])
            prompt_parts.extend([
                f"Incorporate the following object images as assets: {object_filenames_str}.",
                f"These images are in the '{object_image_folder}' directory.",
                "Use simple filename references in the image attribute, like: image=cat.png",
                "Do NOT use base64 encoding - just use the filename directly.",
                "The image paths will be processed later to embed the actual image data."
            ])

        prompt_parts.extend([
            "Position and size elements based on their approximate location in the original image.",
            "Create complete Draw.io XML structure with proper mxGraphModel, root, and mxCell elements.",
            "Ensure all mxCell elements have unique id attributes."
        ])
        
        user_prompt = " ".join(prompt_parts)

        # System instructions semplificato per riferimenti diretti
        simple_ref_instructions = """
You are an expert Draw.io diagram generator.
Create Draw.io XML using simple filename references for images.

Structure:
<mxfile compressed="false" host="GeminiAgent" version="1.0" type="device">
  <diagram id="diagram-1" name="Page-1">
    <mxGraphModel dx="1000" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <mxCell id="obj_1" value="object_name" style="shape=image;html=1;imageAspect=1;aspect=fixed;image=filename.png" vertex="1" parent="1">
          <mxGeometry x="100" y="100" width="80" height="60" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>

Use simple filename references like 'image=cat.png' - do NOT embed base64 data.
Position elements to match the original image layout.
"""

        response = client.models.generate_content(
            contents=[user_prompt, original_image],
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=simple_ref_instructions,
                temperature=0,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                ]
            )
        )

        xml_output = response.text.strip()
        
        # Clean up markdown formatting
        if xml_output.startswith("```xml"): 
            xml_output = xml_output[len("```xml"):]
        if xml_output.endswith("```"): 
            xml_output = xml_output[:-len("```")]
        
        xml_output = xml_output.strip()
        
        # POST-PROCESSING: Sostituisci i riferimenti con base64
        print("Post-processing: Converting image references to base64...")
        final_xml = replace_image_references_xml_parser(xml_output, object_image_folder)
        save_drawio_xml(final_xml, "drawio_output", output_directory="output_llm")

        return True

    except FileNotFoundError:
        return f"Errore: File immagine originale non trovato a {original_image_path}."
    except Exception as e:
        print(f"Errore dettagliato in generate_drawio_from_image_and_objects_v4: {e}")
        return f"Errore durante la generazione dell'XML Draw.io: {str(e)}"

# Funzione standalone per post-processare XML esistenti
def post_process_drawio_xml_file(xml_file_path: str, base_folder: str = "output_llm", output_path: str = None) -> str:
    """
    Post-processa un file XML Draw.io esistente per sostituire i riferimenti alle immagini
    
    Args:
        xml_file_path: Percorso del file XML Draw.io
        base_folder: Cartella base per le immagini
        output_path: Percorso di output (se None, sovrascrive l'originale)
        
    Returns:
        Percorso del file processato
    """
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        processed_xml = replace_image_references_xml_parser(xml_content, base_folder)
        
        if output_path is None:
            output_path = xml_file_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_xml)
        
        print(f"Processed XML saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing XML file: {e}")
        return xml_file_path
    

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