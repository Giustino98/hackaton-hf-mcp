import gradio as gr
import os
import shutil
import uuid
import urllib.parse
import base64
import requests
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage
from nodes.core import graph

# --- Configuration & Setup ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "files")
OUTPUT_LLM_DIR = os.path.join(BASE_DIR, "output_llm")

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LLM_DIR, exist_ok=True)

# Initialize Langfuse CallbackHandler
langfuse_handler = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        print("✅ Langfuse Tracing ENABLED.")
    except Exception as e:
        print(f"❌ Failed to initialize Langfuse Handler: {e}. Tracing will be disabled.")
else:
    print("⚠️ Langfuse Tracing DISABLED (API keys or host not fully configured).")

# CSS semplificato
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
}

.main-title {
    text-align: center;
    color: #1f2937;
    margin-bottom: 2rem;
}

.section-header {
    color: #374151;
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.5rem;
}

.status-success { color: #059669; font-weight: 500; }
.status-error { color: #dc2626; font-weight: 500; }
.status-processing { color: #2563eb; font-weight: 500; }

.drawio-preview {
    border: 2px solid #d1d5db;
    border-radius: 8px;
    background: #f9fafb;
    padding: 1rem;
    text-align: center;
}

.xml-container {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}
"""

def xml_to_svg_image(xml_content):
    """Converte XML draw.io in immagine SVG usando l'API di diagrams.net"""
    if not xml_content or xml_content.startswith("Error"):
        return None
    
    try:
        # Usa l'API pubblica di diagrams.net per convertire XML in SVG
        url = "https://convert.diagrams.net/convert"
        
        data = {
            'format': 'svg',
            'xml': xml_content,
            'bg': 'white'
        }
        
        response = requests.post(url, data=data, timeout=30)
        
        if response.status_code == 200:
            # CORREZIONE: Usa un nome file unico per evitare conflitti
            svg_filename = f"diagram_{uuid.uuid4().hex[:8]}.svg"
            svg_path = os.path.join(OUTPUT_LLM_DIR, svg_filename)
            
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return svg_path
        else:
            print(f"⚠️ Errore API conversione: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Errore conversione SVG: {str(e)}")
        return None

def create_drawio_preview(xml_content):
    """Crea un'anteprima del diagramma e restituisce HTML + path SVG"""
    if not xml_content or xml_content.startswith("Error"):
        preview_html = """
        <div class="drawio-preview">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #6b7280;">Nessun diagramma generato</h3>
            <p style="color: #9ca3af;">Carica un'immagine e genera il diagramma per visualizzarlo qui.</p>
        </div>
        """
        return preview_html, None
    
    # Prova a convertire in immagine
    svg_path = xml_to_svg_image(xml_content)
    
    if svg_path and os.path.exists(svg_path):
        # CORREZIONE: Restituisci sia l'HTML che il path del file SVG
        preview_html = f"""
        <div class="drawio-preview">
            <h3 style="color: #059669; margin-bottom: 1rem;">✅ Diagramma Generato con Successo!</h3>
            <p style="color: #6b7280; margin-bottom: 1rem;">
                Il diagramma è stato convertito e può essere scaricato come file .drawio
            </p>
            <p style="color: #374151; font-size: 0.9rem;">
                💡 <strong>Suggerimento:</strong> Scarica il file .drawio e aprilo in 
                <a href="https://app.diagrams.net/" target="_blank" style="color: #2563eb;">diagrams.net</a> 
                per modificarlo
            </p>
        </div>
        """
        return preview_html, svg_path
    else:
        preview_html = f"""
        <div class="drawio-preview">
            <h3 style="color: #059669; margin-bottom: 1rem;">✅ XML Draw.io Generato!</h3>
            <p style="color: #6b7280;">
                Il diagramma è stato generato come codice XML. 
                Puoi copiare il codice qui sotto o scaricare il file .drawio.
            </p>
            <div style="margin: 1rem 0; padding: 1rem; background: #fef3c7; border-radius: 6px; border-left: 4px solid #f59e0b;">
                <strong>📋 Come utilizzarlo:</strong><br>
                1. Scarica il file .drawio qui sotto<br>
                2. Vai su <a href="https://app.diagrams.net/" target="_blank" style="color: #2563eb;">diagrams.net</a><br>
                3. Carica il file per visualizzare e modificare il diagramma
            </div>
        </div>
        """
        return preview_html, None

def process_image_to_drawio(uploaded_image_temp_path, progress=gr.Progress()):
    """Esegue il processo di conversione dell'immagine in un diagramma Draw.io.
    Arguments:
        uploaded_image_temp_path (str): Percorso dell'immagine caricata
        
    Returns:
        tuple: Contiene il percorso dell'immagine, messaggio di stato, contenuto XML Draw.io."""
    if uploaded_image_temp_path is None:
        preview_html, svg_path = create_drawio_preview(None)
        return (
            None, 
            "❌ Per favore carica un'immagine.", 
            None, 
            None,
            preview_html,
            None  # SVG file per visualizzazione
        )

    progress(0.1, desc="Inizializzazione...")
    
    agent_input_image_path = None
    fixed_tool_output_filename = "drawio_output.drawio"
    generated_drawio_path = os.path.join(OUTPUT_LLM_DIR, fixed_tool_output_filename)

    status_message = "🚀 Inizializzazione..."
    drawio_xml_content = None
    downloadable_file_path = None

    try:
        progress(0.2, desc="Preparazione immagine...")
        original_extension = os.path.splitext(os.path.basename(uploaded_image_temp_path))[1]
        if not original_extension: 
            original_extension = ".png"
        unique_filename = f"{uuid.uuid4()}{original_extension}"
        agent_input_image_path = os.path.join(FILES_DIR, unique_filename)

        shutil.copy(uploaded_image_temp_path, agent_input_image_path)
        status_message = "📁 Immagine preparata per l'elaborazione"
        print(f"✅ Copied uploaded image to: {agent_input_image_path}")

        progress(0.3, desc="Configurazione agente...")
        message_file_path_component = os.path.join("files", unique_filename)
        question_text = "Generate the drawio diagram for the provided image."
        messages = [HumanMessage(content=f"{question_text} Path: {message_file_path_component}")]

        if os.path.exists(generated_drawio_path):
            os.remove(generated_drawio_path)

        progress(0.5, desc="Elaborazione AI in corso...")
        status_message = "🤖 Elaborazione AI in corso... Attendere prego"
        print(f"🔄 Invoking agent with message: {messages[0].content}")

        invoke_config = {}
        if langfuse_handler:
            invoke_config["callbacks"] = [langfuse_handler]

        progress(0.7, desc="Generazione diagramma...")
        agent_response = graph.invoke(input={"messages": messages}, config=invoke_config)
        
        progress(0.9, desc="Finalizzazione...")
        agent_final_text_response = "Elaborazione completata"
        if agent_response and "messages" in agent_response and agent_response["messages"]:
            content = agent_response["messages"][-1].content
            if isinstance(content, list):
                agent_final_text_response = content[-1] if content and isinstance(content[-1], str) else "Elaborazione completata"
            elif isinstance(content, str):
                agent_final_text_response = content
        
        if os.path.exists(generated_drawio_path):
            with open(generated_drawio_path, "r", encoding="utf-8") as f:
                drawio_xml_content = f.read()
            status_message = f"✅ {agent_final_text_response}\n📊 Diagramma Draw.io generato con successo!"
            downloadable_file_path = generated_drawio_path
            print(f"✅ Successfully read Draw.io XML from: {generated_drawio_path}")
        else:
            error_detail = "File di output non trovato"
            drawio_xml_content = f"Error: {error_detail}"
            status_message = f"❌ Errore: {error_detail}"
            print(f"❌ Error: Draw.io output file NOT FOUND at: {generated_drawio_path}")
            
        progress(1.0, desc="Completato!")
            
    except Exception as e:
        error_msg = f"Si è verificato un errore: {str(e)}"
        print(f"💥 {error_msg}")
        status_message = f"❌ {error_msg}"
        drawio_xml_content = f"Errore: {str(e)}"
    finally:
        if agent_input_image_path and os.path.exists(agent_input_image_path):
            try:
                os.remove(agent_input_image_path)
                print(f"🧹 Cleaned up input image: {agent_input_image_path}")
            except OSError as e_remove:
                print(f"⚠️ Warning: Could not remove temp input file {agent_input_image_path}: {e_remove}")

    # CORREZIONE: Ottieni sia HTML che SVG path
    preview_html, svg_file_path = create_drawio_preview(drawio_xml_content)
    
    return (
        uploaded_image_temp_path, 
        status_message, 
        drawio_xml_content, 
        downloadable_file_path,
        preview_html,
        svg_file_path  # NUOVO: path del file SVG per visualizzazione
    )

# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Default(),
    css=custom_css,
    title="Image to Draw.io Converter"
) as demo:
    
    gr.HTML('<h1 class="main-title">🖼️ ➡️ 📊 Image to Draw.io Converter</h1>')
    gr.Markdown("Carica un'immagine e genera automaticamente un diagramma Draw.io utilizzando l'intelligenza artificiale.")

    with gr.Row():
        # Colonna sinistra - Input
        with gr.Column(scale=1):
            gr.Markdown('<div class="section-header">📤 Carica Immagine</div>')
            input_image = gr.Image(
                type="filepath", 
                label="Seleziona un'immagine",
                height=250
            )
            
            generate_button = gr.Button(
                "🚀 Genera Diagramma", 
                variant="primary",
                size="lg"
            )
            
            gr.Markdown('<div class="section-header">📊 Stato</div>')
            status_output = gr.Textbox(
                value="In attesa di un'immagine...",
                label="",
                interactive=False,
                lines=3,
                show_label=False
            )
        
        # Colonna destra - Preview immagine
        with gr.Column(scale=1):
            gr.Markdown('<div class="section-header">🖼️ Anteprima</div>')
            output_image_display = gr.Image(
                label="", 
                interactive=False, 
                height=250,
                show_label=False
            )

    # Sezione risultati
    gr.Markdown('<div class="section-header">🎯 Risultato</div>')
    
    # CORREZIONE: Layout migliorato per visualizzare SVG
    with gr.Row():
        with gr.Column(scale=2):
            drawio_preview = gr.HTML(
                value=create_drawio_preview(None)[0],  # Solo l'HTML
                show_label=False
            )
        with gr.Column(scale=1):
            # NUOVO: Componente per visualizzare l'SVG generato
            svg_display = gr.Image(
                label="Anteprima Diagramma",
                interactive=False,
                height=300
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**💾 Download**")
            download_drawio_file = gr.File(
                label="File .drawio", 
                interactive=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("**📄 Codice XML**")
            output_xml = gr.Textbox(
                label="", 
                lines=6, 
                interactive=False, 
                show_copy_button=True,
                placeholder="Il codice XML apparirà qui dopo la generazione...",
                show_label=False,
                elem_classes=["xml-container"]
            )

    # CORREZIONE: Event handler aggiornato con nuovo output
    generate_button.click(
        fn=process_image_to_drawio,
        inputs=[input_image],
        outputs=[
            output_image_display, 
            status_output, 
            output_xml, 
            download_drawio_file, 
            drawio_preview,
            svg_display  # NUOVO output per visualizzare l'SVG
        ]
    )

    # Footer semplice
    gr.Markdown("""
    ---
    **ℹ️ Istruzioni:**
    1. Carica un'immagine (diagramma, schema, grafico)
    2. Clicca "Genera Diagramma" 
    3. Scarica il file .drawio o copia l'XML
    4. Apri il file su [diagrams.net](https://app.diagrams.net/) per modificarlo
    """)

if __name__ == "__main__":
    print("🚀 Starting Gradio App for Image to Draw.io conversion...")
    
    if not os.path.exists(FILES_DIR): 
        os.makedirs(FILES_DIR)
    if not os.path.exists(OUTPUT_LLM_DIR): 
        os.makedirs(OUTPUT_LLM_DIR)
    
    print(f"📁 Script base directory: {BASE_DIR}")
    print(f"📤 Directory for uploaded files: {FILES_DIR}")
    print(f"📥 Directory for agent outputs: {OUTPUT_LLM_DIR}")
    demo.launch(
        debug=True, 
        share=False,
        show_api=True,
        mcp_server=True
    )