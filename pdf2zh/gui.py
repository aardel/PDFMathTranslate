import asyncio
import cgi
import os
import shutil
import tempfile
import uuid
from asyncio import CancelledError
from pathlib import Path
from urllib.parse import quote
import typing as T

import gradio as gr
import requests
import tqdm
from string import Template
import logging
import re

from pdf2zh.high_level import translate
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.kernel import KernelRegistry
from pdf2zh.translator import (
    BaseTranslator,
    BingTranslator,
    DeepLXTranslator,
    ArgosTranslator,
    GoogleTranslator,
    OllamaTranslator,
    XinferenceTranslator,
    GroqTranslator,
)
from babeldoc.docvision.doclayout import OnnxModel

logger = logging.getLogger(__name__)


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = OnnxModel.load_available()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)


BABELDOC_MODEL = _LazyModel()
PREVIEW_ROOT = Path(tempfile.gettempdir()) / "pdf2zh-previews"


def pdf_preview_html(file_path):
    """Render a server-hosted PDF with the browser's built-in viewer."""
    if not file_path:
        return '<div class="preview-placeholder">Upload a PDF to preview it.</div>'
    source = Path(file_path).resolve()
    preview_root = PREVIEW_ROOT.resolve()
    try:
        source.relative_to(preview_root)
        preview_file = source
    except ValueError:
        preview_dir = preview_root / uuid.uuid4().hex
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_file = preview_dir / source.name
        shutil.copy2(source, preview_file)
    normalized_path = str(preview_file).replace("\\", "/")
    file_url = "/gradio_api/file=" + quote(normalized_path, safe="/:")
    return (
        f'<iframe class="native-pdf-preview" src="{file_url}" '
        'title="Document Preview"></iframe>'
    )


def prepare_pdf_preview(file_path):
    """Validate and stage an uploaded PDF in the inline-preview directory."""
    if not file_path:
        return pdf_preview_html(None)
    source = Path(file_path)
    with source.open("rb") as selected_file:
        if selected_file.read(4) != b"%PDF":
            raise gr.Error("The selected file is not a valid PDF")
    return pdf_preview_html(source)


LANGUAGE_MARKERS = {
    "German": {"der", "die", "das", "und", "ist", "mit", "für", "von", "auf", "ein", "eine", "nicht"},
    "English": {"the", "and", "is", "with", "for", "from", "this", "that", "not", "are", "of", "to"},
    "French": {"le", "la", "les", "et", "est", "avec", "pour", "des", "une", "dans", "pas", "du"},
    "Spanish": {"el", "la", "los", "las", "y", "es", "con", "para", "una", "del", "por", "no"},
    "Italian": {"il", "la", "gli", "le", "e", "è", "con", "per", "una", "del", "non", "di"},
}


def detect_document_language(text: str) -> tuple[str, str]:
    """Detect a supported source language without a network dependency."""
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese", "high"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean", "high"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Simplified Chinese", "high"
    if re.search(r"[\u0400-\u04ff]", text):
        return "Russian", "high"

    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    scores = {
        language: sum(word in markers for word in words)
        for language, markers in LANGUAGE_MARKERS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    if language == "German":
        score += len(re.findall(r"[äöüß]", text.lower()))
    confidence = "high" if score >= 12 else "medium" if score >= 4 else "low"
    return language, confidence


def analyze_pdf_upload(file_path):
    """Build the preview and recommend settings from PDF content/layout."""
    preview_html = prepare_pdf_preview(file_path)
    source = Path(str(file_path)).resolve()

    import pymupdf

    with pymupdf.open(source) as document:
        sampled_pages = min(document.page_count, 10)
        text_parts = []
        font_names = set()
        toc_entries = 0
        code_lines = 0
        dense_pages = 0
        scanned_pages = 0

        for page_number in range(sampled_pages):
            page = document[page_number]
            page_text = page.get_text("text")
            text_parts.append(page_text)
            toc_entries += len(
                re.findall(r"\.{5,}\s*\d+\s*$", page_text, flags=re.MULTILINE)
            )
            code_lines += len(
                re.findall(r"^\s*N\d+\s+(?:G|M)\d+", page_text, flags=re.MULTILINE)
            )
            if len(page.get_text("blocks")) >= 35:
                dense_pages += 1
            if len(page_text.strip()) < 50 and page.get_images(full=True):
                scanned_pages += 1
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("font"):
                            font_names.add(span["font"])

        page_count = document.page_count

    detected_language, confidence = detect_document_language("\n".join(text_parts))
    risks = []
    recommendations = []
    if toc_entries >= 3:
        risks.append(f"table of contents / dot leaders ({toc_entries} entries)")
        recommendations.append("TOC protection: enabled")
    if code_lines >= 3:
        risks.append(f"preformatted/code listing ({code_lines} lines)")
        recommendations.append("Code-line positioning protection: enabled")
    if len(font_names) >= 6:
        risks.append(f"mixed typography ({len(font_names)} fonts)")
    if dense_pages:
        risks.append(f"dense positioned text ({dense_pages} sampled page(s))")
    if scanned_pages:
        risks.append(f"image-only/scanned content ({scanned_pages} sampled page(s))")
        recommendations.append("OCR is required for scanned pages")

    available_modes = KernelRegistry.available()
    recommended_mode = "precise" if risks and "precise" in available_modes else "fast"
    if recommended_mode == "fast" and risks:
        recommendations.append("Translation Mode: fast (layout protection is automatic)")
    else:
        recommendations.append(f"Mode: {recommended_mode}")
    recommendations.append("Skip font subsetting: off")

    risk_summary = ", ".join(risks) if risks else "no major layout risks detected"
    report = (
        "### Document analysis\n"
        f"- **Detected source:** {detected_language} ({confidence} confidence)\n"
        f"- **Document:** {page_count} page(s); {risk_summary}\n"
        f"- **Recommended:** {'; '.join(recommendations)}"
    )
    return (
        preview_html,
        gr.update(value=report, visible=True),
        gr.update(value=detected_language),
        gr.update(value=recommended_mode),
    )


# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "Argos Translate": ArgosTranslator,
    "Groq": GroqTranslator,
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
else:
    enabled_services = list(service_map.keys())


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    mode_choice,
    recaptcha_response,
    state,
    progress=gr.Progress(track_tqdm=True),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0.01, desc="Preparing translation...")

    output = Path("pdf2zh_files") / str(session_id)
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        if Path(file_input).suffix.lower() != ".pdf":
            raise gr.Error("Only PDF files are supported")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    with open(file_path, "rb") as selected_file:
        is_pdf = selected_file.read(4) == b"%PDF"
    if not is_pdf:
        raise gr.Error("The selected file is not a valid PDF")

    progress(0.04, desc="PDF validated. Loading translation resources...")

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    if service not in service_map:
        raise gr.Error("Unsupported translation service")
    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        try:
            for p in page_input.split(","):
                if "-" in p:
                    start, end = p.split("-", 1)
                    selected_page.extend(range(int(start) - 1, int(end)))
                else:
                    selected_page.append(int(p) - 1)
        except (AttributeError, ValueError):
            raise gr.Error("Invalid page range. Use values such as 1,3-5")
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(update):
        if isinstance(update, dict):
            desc = update.get("stage") or "Translating..."
            fraction = float(
                update.get("overall_progress", update.get("stage_progress", 0.0))
            )
            if fraction > 1:
                fraction /= 100
        else:
            desc = getattr(update, "desc", None) or "Translating..."
            total = getattr(update, "total", 0) or 0
            fraction = (getattr(update, "n", 0) / total) if total else 0.0
        progress(max(0.0, min(1.0, fraction)), desc=desc)

    try:
        threads = int(threads)
    except (TypeError, ValueError):
        threads = 1

    try:
        from pdf2zh.kernel.protocol import TranslateRequest

        available_modes = KernelRegistry.available()
        if mode_choice not in available_modes:
            raise RuntimeError(
                f"Translation mode '{mode_choice}' is not installed. "
                f"Available mode: {', '.join(available_modes)}"
            )
        KernelRegistry.switch(mode_choice)
        kernel = KernelRegistry.get()
        request = TranslateRequest(
            files=[str(file_raw)],
            output=str(output),
            pages=selected_page,
            lang_in=lang_from,
            lang_out=lang_to,
            service=f"{translator.name}",
            thread=int(threads),
            envs=_envs,
            prompt=str(prompt) if prompt else None,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
        )
        progress(0.08, desc="Analyzing PDF layout...")
        results = kernel.translate(
            request,
            callback=progress_bar,
            cancellation_event=cancellation_event_map[session_id],
        )
        if results:
            if results[0].mono_pdf:
                file_mono = Path(results[0].mono_pdf)
            if results[0].dual_pdf:
                file_dual = Path(results[0].dual_pdf)
    except CancelledError:
        raise gr.Error("Translation cancelled")
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Translation failed")
        raise gr.Error(
            f"Translation failed ({type(exc).__name__}): {exc}"
        ) from exc
    finally:
        cancellation_event_map.pop(session_id, None)
    print(f"Files after translation: {os.listdir(output)}")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        pdf_preview_html(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        ArgosTranslator,
        GroqTranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }

    .native-pdf-preview {
        width: 100%;
        height: 2000px;
        border: 0;
        border-radius: 8px;
        background: white;
    }

    .preview-placeholder {
        min-height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #777;
        border: 1px dashed #bbb;
        border-radius: 8px;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title="PDF Translator",
    theme=gr.themes.Default(
        primary_hue=custom_blue,
        spacing_size="md",
        radius_size="lg",
        font=(
            gr.themes.Font("Segoe UI"),
            gr.themes.Font("Arial"),
            gr.themes.Font("sans-serif"),
        ),
        font_mono=(
            gr.themes.Font("Consolas"),
            gr.themes.Font("monospace"),
        ),
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    gr.Markdown("# PDF Translator")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## File | < 5 MB" if flag_demo else "## File")
            file_type = gr.Radio(
                choices=["File", "Link"],
                label="Type",
                value="File",
            )
            file_input = gr.File(
                label="File",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["input-file"],
            )
            analysis_report = gr.Markdown(visible=False)
            link_input = gr.Textbox(
                label="Link",
                visible=False,
                interactive=True,
            )
            gr.Markdown("## Option")
            service = gr.Dropdown(
                label="Service",
                choices=enabled_services,
                value=enabled_services[0],
            )
            envs = []
            for i in range(3):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label="Translate from",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_FROM", "English"),
                )
                lang_to = gr.Dropdown(
                    label="Translate to",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese"),
                )
            page_range = gr.Radio(
                choices=page_map.keys(),
                label="Pages",
                value=list(page_map.keys())[0],
            )

            page_input = gr.Textbox(
                label="Page range",
                visible=False,
                interactive=True,
            )

            with gr.Accordion("Open for More Experimental Options!", open=False):
                gr.Markdown("#### Experimental")
                threads = gr.Textbox(
                    label="number of threads", interactive=True, value="4"
                )
                skip_subset_fonts = gr.Checkbox(
                    label="Skip font subsetting", interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label="Ignore cache", interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label="Custom formula font regex (vfont)",
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label="Custom Prompt for llm", interactive=True, visible=False
                )
                mode_choices = KernelRegistry.available()
                mode_choice = gr.Dropdown(
                    label="Translation Mode",
                    choices=mode_choices,
                    value="fast" if "fast" in mode_choices else mode_choices[0],
                    interactive=True,
                )
                envs.append(prompt)

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                _envs = []
                for i in range(4):
                    _envs.append(gr.update(visible=False, value=""))
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                return _envs

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            output_title = gr.Markdown("## Translated", visible=False)
            output_file_mono = gr.File(
                label="Download Translation (Mono)", visible=False
            )
            output_file_dual = gr.File(
                label="Download Translation (Dual)", visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            translate_btn = gr.Button("Translate", variant="primary")
            cancellation_btn = gr.Button("Cancel", variant="secondary")
            page_range.select(
                on_select_page,
                page_range,
                page_input,
                queue=False,
                show_progress="hidden",
            )
            service.select(
                on_select_service,
                service,
                envs,
                queue=False,
                show_progress="hidden",
            )
            vfont.change(
                on_vfont_change,
                inputs=vfont,
                outputs=None,
                queue=False,
                show_progress="hidden",
            )
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                queue=False,
                show_progress="hidden",
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("## Preview")
            preview = gr.HTML(
                value=pdf_preview_html(None),
                label="Document Preview",
                visible=True,
            )

    # Event handlers
    file_input.upload(
        analyze_pdf_upload,
        inputs=file_input,
        outputs=[preview, analysis_report, lang_from, mode_choice],
        queue=False,
        show_progress="hidden",
    )

    state = gr.State({"session_id": None})

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            mode_choice,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
    ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
        queue=False,
        show_progress="hidden",
    )


# Translation and other server-side callbacks use Gradio's event queue.
demo.queue(default_concurrency_limit=1)


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)

    auth_kwargs = {}
    if len(user_list) > 0:
        auth_kwargs = {"auth": user_list, "auth_message": html}

    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    inline_pdf_paths = [str(PREVIEW_ROOT.resolve())]

    if flag_demo:
        demo.launch(
            server_name="0.0.0.0",
            max_file_size="5mb",
            inbrowser=True,
            allowed_paths=inline_pdf_paths,
        )
        return

    # Ensure Gradio's own localhost health check bypasses system proxies. A
    # single launch keeps the queue worker from being torn down by retries.
    local_hosts = "localhost,127.0.0.1,::1"
    for proxy_bypass_var in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(proxy_bypass_var, "")
        entries = [value for value in (existing, local_hosts) if value]
        os.environ[proxy_bypass_var] = ",".join(entries)

    demo.launch(
        server_name="127.0.0.1",
        debug=True,
        inbrowser=True,
        share=share,
        server_port=server_port,
        allowed_paths=inline_pdf_paths,
        show_error=True,
        **auth_kwargs,
    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
