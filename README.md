# Image to Draw.io Converter

A multi-agent application that converts images into editable [Draw.io](https://app.diagrams.net/) diagrams using advanced LLM-based object detection and diagram generation. The system is built with [LangGraph](https://github.com/langchain-ai/langgraph) and features a modern [Gradio](https://gradio.app/) web interface, which is also exposed as an MCP server for integration with external clients.

---

## Features

- **Multi-Agent Pipeline (LangGraph):**
  - **Supervisor Agent:** Orchestrates the workflow and coordinates the agents.
  - **Object Detection Agent (React):** Uses an LLM to detect and extract objects from the uploaded image.
  - **Draw.io Generator Agent (React):** Converts detected objects into Draw.io XML diagrams.

- **Automated Workflow:**
  1. Upload an image via the Gradio interface.
  2. The object detection agent identifies and extracts key objects.
  3. The diagram generator agent creates a Draw.io XML diagram.
  4. Download the `.drawio` file or preview the SVG directly in the browser.

- **Modern Gradio UI:**
  - Simple drag-and-drop image upload.
  - Real-time status updates and diagram preview.
  - Downloadable Draw.io file and copyable XML.
  - SVG preview of the generated diagram.
  - Exposed as an MCP server for programmatic access.

---

## Requirements

- Python 3.10+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- API keys for your LLM provider (e.g., Google Gemini).
- Configure your `.env` file:
  ```
  GEMINI_API_KEY=your_gemini_api_key
  GEMINI_MODEL_NAME=gemini-2.5-pro-preview-06-05
  GEMINI_THINKING_BUDGET=128
  ```

---

## Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   - Create a `.env` file in the project root with your API keys and model settings.

3. **Start the application:**
   ```sh
   python app.py
   ```

4. **Access the web interface:**
   - Open the provided local URL in your browser.
   - Upload an image (diagram, sketch, chart, etc.).
   - Click "Generate Diagram".
   - Download the `.drawio` file or copy the XML.
   - Open the file in [diagrams.net](https://app.diagrams.net/) for further editing.

---

## MCP Server Integration

The Gradio app is also exposed as an MCP server (`mcp_server=True`), allowing integration with any MCP-compatible client or workflow. This enables automated or remote usage in larger pipelines.

---

## Project Structure

- `app.py` — Main entry point, Gradio UI, and workflow orchestration.
- `nodes/` — Agent and LangGraph node definitions.
- `tools/` — LLM tools for object detection and Draw.io generation.
- `output_llm/` — Generated Draw.io files, SVG previews, and logs.
- `files/` — Uploaded user images.

---

## Notes

- The LLM model and thinking budget are fully configurable via `.env` for maximum flexibility.
- The application is designed for easy extension with new agents or tools.
- Performances can vary a lot based on the LLM: in my experience, Gemini 2.5 pro performed much better than Gemini 2.5 flash.

---

## License

MIT License

---

**Author:** Giustino Esposito