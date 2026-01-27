"""Simple Gradio UI for testing RLM capabilities.

Features:
- Directory picker with native folder browser (local mode)
- File upload with S3 storage (cloud mode)
- Query input
- REAL-TIME thinking display (streams as investigation progresses)
- Citations panel
- Thinking trace panel
"""

import gradio as gr
import asyncio
import threading
import queue
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Generator
import os

from ..core.models import GeminiClient
from ..rlm.engine import RLMEngine, RLMConfig
from ..rlm.state import InvestigationState, ThinkingStep, Citation, StepType
from ..service.config import ServiceConfig

logger = logging.getLogger(__name__)


def get_storage_mode() -> str:
    """Get storage mode from environment."""
    return os.getenv("IRYS_STORAGE_MODE", "local")


def browse_folder() -> str:
    """Open native folder picker dialog and return selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front

        # Open folder picker
        folder_path = filedialog.askdirectory(
            title="Select Document Repository",
            initialdir=os.path.expanduser("~")
        )

        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"Folder picker error: {e}")
        return ""


class RLMApp:
    """Gradio application wrapper for RLM with real-time streaming."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[ServiceConfig] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.config = config or ServiceConfig.from_env()
        self.storage_mode = get_storage_mode()
        self.current_state: Optional[InvestigationState] = None
        self.thinking_log: list[str] = []
        self.citations_log: list[str] = []
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.final_output = ""
        self.error_msg = ""
        self._temp_dirs: dict[str, Path] = {}  # Track temp dirs for cleanup

    def _generate_session_id(self) -> str:
        """Generate a unique session/job ID."""
        return f"ui_{uuid.uuid4().hex[:12]}"

    async def _upload_files_to_s3(
        self,
        files: list[tuple[str, bytes]],
        session_id: str,
    ) -> str:
        """Upload files to S3 and return the S3 prefix.

        Args:
            files: List of (filename, content) tuples
            session_id: Unique session identifier

        Returns:
            S3 prefix where files were uploaded
        """
        from ..service.s3_repository import S3Repository

        s3_repo = S3Repository(
            bucket=self.config.s3_bucket,
            config=self.config,
        )

        prefix = await s3_repo.upload_files(session_id, files)
        logger.info(f"Uploaded {len(files)} files to S3: {prefix}")
        return prefix

    async def _download_s3_to_temp(self, s3_prefix: str, session_id: str) -> Path:
        """Download files from S3 prefix to temp directory for processing.

        Args:
            s3_prefix: S3 prefix containing the files
            session_id: Session ID for temp dir naming

        Returns:
            Path to temp directory with downloaded files
        """
        from ..service.s3_repository import S3Repository

        s3_repo = S3Repository(
            bucket=self.config.s3_bucket,
            prefix=s3_prefix,
            config=self.config,
        )

        temp_dir = await s3_repo.download_to_temp(session_id)
        self._temp_dirs[session_id] = temp_dir
        logger.info(f"Downloaded S3 files to temp: {temp_dir}")
        return temp_dir

    def _save_files_to_temp(
        self,
        files: list[tuple[str, bytes]],
        session_id: str,
    ) -> Path:
        """Save uploaded files to local temp directory.

        Args:
            files: List of (filename, content) tuples
            session_id: Unique session identifier

        Returns:
            Path to temp directory with files
        """
        temp_dir = Path(self.config.temp_dir) / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in files:
            file_path = temp_dir / filename
            file_path.write_bytes(content)
            logger.debug(f"Saved file: {file_path}")

        self._temp_dirs[session_id] = temp_dir
        logger.info(f"Saved {len(files)} files to temp: {temp_dir}")
        return temp_dir

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up temp directory for a session."""
        if session_id in self._temp_dirs:
            import shutil
            temp_dir = self._temp_dirs.pop(session_id)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")

    def on_thinking_step(self, step: ThinkingStep):
        """Callback for thinking steps - pushes to queue for real-time updates."""
        self.thinking_log.append(step.display)
        self.update_queue.put(("thinking", step.display))

    def on_citation(self, citation: Citation):
        """Callback for citations."""
        page_str = f", p. {citation.page}" if citation.page else ""
        # Build full citation with text and context
        citation_parts = [f"[{len(self.citations_log) + 1}] {citation.document}{page_str}"]
        if citation.context:
            citation_parts.append(f"    {citation.context}")
        if citation.text:
            # Show full text (truncate only for display if extremely long)
            text_display = citation.text if len(citation.text) <= 500 else citation.text[:500] + "..."
            citation_parts.append(f"    Text: {text_display}")
        citation_text = "\n".join(citation_parts)
        self.citations_log.append(citation_text)
        self.update_queue.put(("citation", citation_text))

    def run_investigation_async(self, query: str, repo_path: str):
        """Run investigation in background thread."""
        async def _run():
            try:
                client = GeminiClient(api_key=self.api_key)
                engine = RLMEngine(
                    gemini_client=client,
                    config=RLMConfig(),
                    on_step=self.on_thinking_step,
                    on_citation=self.on_citation,
                )
                state = await engine.investigate(query, repo_path)
                self.current_state = state
                self.final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", state))
            except Exception as e:
                self.error_msg = str(e)
                self.update_queue.put(("error", str(e)))

        asyncio.run(_run())

    def run_investigation_with_upload_async(
        self,
        query: str,
        files: list[tuple[str, bytes]],
        session_id: str,
    ):
        """Run investigation with uploaded files in background thread."""
        async def _run():
            repo_path = None
            try:
                # Handle file storage based on mode
                if self.storage_mode != "local":
                    # S3 mode: upload files then download to temp
                    s3_prefix = await self._upload_files_to_s3(files, session_id)
                    repo_path = await self._download_s3_to_temp(s3_prefix, session_id)
                else:
                    # Local mode: save directly to temp
                    repo_path = self._save_files_to_temp(files, session_id)

                # Run investigation
                client = GeminiClient(api_key=self.api_key)
                engine = RLMEngine(
                    gemini_client=client,
                    config=RLMConfig(),
                    on_step=self.on_thinking_step,
                    on_citation=self.on_citation,
                )
                state = await engine.investigate(query, str(repo_path))
                self.current_state = state
                self.final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", state))
            except Exception as e:
                self.error_msg = str(e)
                self.update_queue.put(("error", str(e)))
            finally:
                # Cleanup temp directory
                self._cleanup_session(session_id)

        asyncio.run(_run())

    def stream_investigation_with_upload(
        self,
        query: str,
        uploaded_files: list,
    ) -> Generator[tuple, None, None]:
        """
        Generator that streams investigation with uploaded files.
        Yields: (output, thinking_trace, citations, status)

        Args:
            query: The investigation query
            uploaded_files: List of Gradio file objects
        """
        self.thinking_log = []
        self.citations_log = []
        self.final_output = ""
        self.error_msg = ""
        self.update_queue = queue.Queue()

        if not uploaded_files:
            yield ("", "", "", "Error: Please upload at least one document")
            return

        if not query.strip():
            yield ("", "", "", "Error: Please enter a query")
            return

        # Convert Gradio files to (filename, bytes) tuples
        files: list[tuple[str, bytes]] = []
        for file in uploaded_files:
            try:
                filename = Path(file.name).name
                with open(file.name, "rb") as f:
                    content = f.read()
                files.append((filename, content))
            except Exception as e:
                logger.error(f"Error reading file {file.name}: {e}")
                yield ("", "", "", f"Error reading file: {e}")
                return

        session_id = self._generate_session_id()
        logger.info(f"Starting investigation session {session_id} with {len(files)} files")

        # Start investigation in background thread
        self.is_running = True
        thread = threading.Thread(
            target=self.run_investigation_with_upload_async,
            args=(query, files, session_id)
        )
        thread.start()

        # Use the common streaming loop
        yield from self._stream_updates(thread)

    def stream_investigation(
        self,
        query: str,
        repo_path: str,
    ) -> Generator[tuple, None, None]:
        """
        Generator that streams investigation updates in real-time.
        Yields: (output, thinking_trace, citations, status)
        """
        self.thinking_log = []
        self.citations_log = []
        self.final_output = ""
        self.error_msg = ""
        self.update_queue = queue.Queue()

        if not repo_path or not Path(repo_path).exists():
            yield ("", "", "", "Error: Please select a valid directory")
            return

        if not query.strip():
            yield ("", "", "", "Error: Please enter a query")
            return

        # Start investigation in background thread
        self.is_running = True
        thread = threading.Thread(
            target=self.run_investigation_async,
            args=(query, repo_path)
        )
        thread.start()

        # Use the common streaming loop
        yield from self._stream_updates(thread)

    def _stream_updates(self, thread: threading.Thread) -> Generator[tuple, None, None]:
        """Common streaming loop for investigation updates."""

        start_time = time.time()

        # Stream updates as they come in
        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.5)

                if update_type == "thinking":
                    elapsed = time.time() - start_time
                    status = (
                        f"Status: INVESTIGATING...\n"
                        f"Elapsed: {elapsed:.1f}s\n"
                        f"Steps: {len(self.thinking_log)}\n"
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "Finding sources...",
                        status
                    )

                elif update_type == "citation":
                    # Just update citations
                    pass

                elif update_type == "complete":
                    self.is_running = False
                    state = data
                    elapsed = time.time() - start_time
                    status = (
                        f"Status: COMPLETE\n"
                        f"Documents read: {state.documents_read}\n"
                        f"Searches performed: {state.searches_performed}\n"
                        f"Citations found: {len(state.citations)}\n"
                        f"Duration: {elapsed:.1f}s"
                    )
                    yield (
                        self.final_output,
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "No citations found",
                        status
                    )
                    return

                elif update_type == "error":
                    self.is_running = False
                    yield (
                        "",
                        "\n".join(self.thinking_log),
                        "",
                        f"Error: {data}"
                    )
                    return

            except queue.Empty:
                # No update, yield current state to keep UI responsive
                elapsed = time.time() - start_time
                if len(self.thinking_log) > 0:
                    status = (
                        f"Status: INVESTIGATING...\n"
                        f"Elapsed: {elapsed:.1f}s\n"
                        f"Steps: {len(self.thinking_log)}\n"
                        f"Citations: {len(self.citations_log)}"
                    )
                    yield (
                        "*Investigation in progress...*",
                        "\n".join(self.thinking_log),
                        "\n".join(self.citations_log) or "Finding sources...",
                        status
                    )

        # Final yield after thread completes
        thread.join()


def create_app(api_key: Optional[str] = None) -> gr.Blocks:
    """Create the Gradio application with real-time streaming.

    The UI adapts based on IRYS_STORAGE_MODE:
    - "local": Shows folder browser for local file system
    - "s3" or other: Shows file upload component for cloud storage
    """
    app = RLMApp(api_key=api_key)
    storage_mode = get_storage_mode()
    is_local_mode = storage_mode == "local"

    with gr.Blocks(
        title="Irys RLM - Legal Document Analysis",
    ) as demo:
        gr.Markdown("# Irys RLM - Recursive Legal Document Analysis")

        if is_local_mode:
            gr.Markdown(
                "Select a matter repository and ask a legal question. "
                "**Watch the thinking trace update in real-time as the system investigates!**"
            )
        else:
            gr.Markdown(
                "Upload your legal documents and ask a question. "
                "**Watch the thinking trace update in real-time as the system investigates!**"
            )

        with gr.Row():
            with gr.Column(scale=2):
                if is_local_mode:
                    # Local mode: folder browser
                    with gr.Row():
                        repo_path = gr.Textbox(
                            label="Repository Path",
                            placeholder="Enter path or click Browse...",
                            info="Full path to the folder containing legal documents",
                            value="",
                            scale=4,
                        )
                        browse_btn = gr.Button("Browse", scale=1)
                else:
                    # Cloud mode: file upload
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".rtf", ".mht"],
                        type="filepath",
                    )
                    gr.Markdown(
                        "*Supported formats: PDF, DOCX, DOC, TXT, RTF, MHT*",
                    )

                query = gr.Textbox(
                    label="Legal Query",
                    placeholder="What would you like to investigate?",
                    lines=3,
                )
                submit_btn = gr.Button("Investigate", variant="primary", size="lg")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", lines=8, interactive=False)

        with gr.Tabs():
            with gr.TabItem("Thinking Trace (LIVE)"):
                thinking = gr.Textbox(
                    label="Thinking Steps - Updates in real-time!",
                    lines=30,
                    interactive=False,
                    autoscroll=True,
                )

            with gr.TabItem("Analysis Output"):
                output = gr.Markdown(label="Analysis")

            with gr.TabItem("Citations"):
                citations = gr.Textbox(
                    label="Citations & Sources",
                    lines=25,
                    interactive=False,
                )

        # Wire up buttons based on mode
        if is_local_mode:
            # Local mode: folder browser
            browse_btn.click(
                fn=browse_folder,
                inputs=[],
                outputs=[repo_path],
            )
            submit_btn.click(
                fn=app.stream_investigation,
                inputs=[query, repo_path],
                outputs=[output, thinking, citations, status],
            )
        else:
            # Cloud mode: file upload
            submit_btn.click(
                fn=app.stream_investigation_with_upload,
                inputs=[query, file_upload],
                outputs=[output, thinking, citations, status],
            )

        # Example queries (click to populate query field)
        gr.Markdown("### Example Queries (click to use)")
        gr.Examples(
            examples=[
                ["What are the key claims in this dispute?"],
                ["What was the initial cost estimate vs actual cost?"],
                ["What damages are being claimed and what is the basis?"],
                ["What is the timeline of events in this case?"],
                ["Who are the key parties and witnesses?"],
            ],
            inputs=[query],
        )

    return demo


def main():
    """Run the application."""
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Server hostname to bind to (default: 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Path to SSL certificate file for HTTPS",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Path to SSL key file for HTTPS",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="Root path for reverse proxy setups (e.g., /app)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No GEMINI_API_KEY provided")

    storage_mode = get_storage_mode()
    print(f"[INFO] Storage mode: {storage_mode}")

    app = create_app(api_key=api_key)

    # Build launch kwargs
    launch_kwargs = {
        "server_port": args.port,
        "server_name": args.server_name,
        "share": args.share,
    }

    # Add SSL if provided
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
        print(f"[INFO] SSL enabled with cert: {args.ssl_certfile}")

    # Add root path for reverse proxy
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path
        print(f"[INFO] Root path: {args.root_path}")

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
