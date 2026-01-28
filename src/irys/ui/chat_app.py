"""Multi-turn Chat UI for Irys RLM.

Features:
- Chat interface with conversation history
- Multiple queries about the same repository
- Thinking logs stored separately (not sent to AI)
- Only user messages + AI responses used for context
"""

import gradio as gr
import asyncio
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field
import os
import json
from datetime import datetime

from ..core.models import GeminiClient
from ..rlm.engine import RLMEngine, RLMConfig
from ..rlm.state import InvestigationState, ThinkingStep, Citation, StepType


@dataclass
class ChatSession:
    """Represents a chat session with a repository."""
    repo_path: str
    conversation: list[tuple[str, str]] = field(default_factory=list)  # (user_msg, ai_msg)
    thinking_logs: list[dict] = field(default_factory=list)  # Per-turn thinking logs
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_turn(self, user_msg: str, ai_msg: str, thinking: list[str], citations: list[str]):
        """Add a conversation turn."""
        self.conversation.append((user_msg, ai_msg))
        self.thinking_logs.append({
            "turn": len(self.conversation),
            "query": user_msg,
            "thinking": thinking,
            "citations": citations,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_for_ai(self, max_turns: int = 5) -> list[dict]:
        """Get conversation history formatted for AI context.

        Only includes user messages and AI responses, not thinking logs.
        """
        history = []
        # Get last N turns
        recent = self.conversation[-max_turns:] if len(self.conversation) > max_turns else self.conversation

        for user_msg, ai_msg in recent:
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "model", "content": ai_msg})

        return history

    def get_thinking_for_turn(self, turn: int) -> Optional[dict]:
        """Get thinking logs for a specific turn."""
        if 0 < turn <= len(self.thinking_logs):
            return self.thinking_logs[turn - 1]
        return None


class ChatApp:
    """Multi-turn chat application for Irys RLM."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.session: Optional[ChatSession] = None
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False

        # Current turn state
        self.current_thinking: list[str] = []
        self.current_citations: list[str] = []

    def on_thinking_step(self, step: ThinkingStep):
        """Callback for thinking steps."""
        self.current_thinking.append(step.display)
        self.update_queue.put(("thinking", step.display))

    def on_citation(self, citation: Citation):
        """Callback for citations."""
        page_str = f", p. {citation.page}" if citation.page else ""
        citation_text = f"[{len(self.current_citations) + 1}] {citation.document}{page_str}"
        if citation.context:
            citation_text += f"\n    {citation.context}"
        self.current_citations.append(citation_text)
        self.update_queue.put(("citation", citation_text))

    def _build_context_prompt(self, query: str) -> str:
        """Build prompt with conversation context."""
        if not self.session or not self.session.conversation:
            return query

        # Get recent conversation history
        history = self.session.get_context_for_ai(max_turns=5)

        if not history:
            return query

        # Format history as context
        context_parts = ["Previous conversation about this matter:"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long responses to save tokens
            content = msg["content"]
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            context_parts.append(f"\n{role}: {content}")

        context_parts.append(f"\n\nCurrent question: {query}")
        context_parts.append("\nAnswer the current question, using context from the previous conversation if relevant.")

        return "\n".join(context_parts)

    def run_investigation_async(self, query: str, repo_path: str, use_context: bool = True):
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

                # Build query with context if we have conversation history
                if use_context and self.session and self.session.conversation:
                    enhanced_query = self._build_context_prompt(query)
                else:
                    enhanced_query = query

                state = await engine.investigate(enhanced_query, repo_path)
                final_output = state.findings.get("final_output", "No output generated")
                self.update_queue.put(("complete", final_output))

            except Exception as e:
                self.update_queue.put(("error", str(e)))

        asyncio.run(_run())

    def chat(
        self,
        message: str,
        history: list[tuple[str, str]],
        repo_path: str,
    ) -> Generator[tuple[list, str, str], None, None]:
        """
        Process a chat message and yield updates.

        Args:
            message: User's message
            history: Gradio chat history
            repo_path: Path to document repository

        Yields:
            (updated_history, thinking_log, status)
        """
        # Validate inputs
        if not repo_path or not Path(repo_path).exists():
            history.append((message, "Error: Please select a valid repository directory."))
            yield history, "", "Error: Invalid repository path"
            return

        if not message.strip():
            yield history, "", "Please enter a question"
            return

        # Initialize or update session
        if self.session is None or self.session.repo_path != repo_path:
            self.session = ChatSession(repo_path=repo_path)

        # Reset current turn state
        self.current_thinking = []
        self.current_citations = []
        self.update_queue = queue.Queue()
        self.is_running = True

        # Add user message to history with placeholder
        history.append((message, None))

        # Start investigation in background
        thread = threading.Thread(
            target=self.run_investigation_async,
            args=(message, repo_path, True)
        )
        thread.start()

        start_time = time.time()
        partial_response = ""

        # Stream updates
        while self.is_running:
            try:
                update_type, data = self.update_queue.get(timeout=0.3)

                if update_type == "thinking":
                    elapsed = time.time() - start_time
                    status = f"Investigating... ({elapsed:.1f}s) - {len(self.current_thinking)} steps"
                    thinking_display = "\n".join(self.current_thinking[-20:])  # Last 20 steps

                    # Update history with "thinking" indicator
                    history[-1] = (message, f"*Investigating... ({len(self.current_thinking)} steps)*")
                    yield history, thinking_display, status

                elif update_type == "citation":
                    pass  # Citations tracked internally

                elif update_type == "complete":
                    self.is_running = False
                    final_output = data
                    elapsed = time.time() - start_time

                    # Store in session
                    self.session.add_turn(
                        user_msg=message,
                        ai_msg=final_output,
                        thinking=self.current_thinking.copy(),
                        citations=self.current_citations.copy()
                    )

                    # Update history with final response
                    history[-1] = (message, final_output)

                    status = f"Complete ({elapsed:.1f}s) - Turn {len(self.session.conversation)}"
                    thinking_display = "\n".join(self.current_thinking)

                    yield history, thinking_display, status
                    return

                elif update_type == "error":
                    self.is_running = False
                    error_msg = f"Error: {data}"
                    history[-1] = (message, error_msg)
                    yield history, "\n".join(self.current_thinking), f"Error: {data}"
                    return

            except queue.Empty:
                # Keep UI responsive
                if self.current_thinking:
                    elapsed = time.time() - start_time
                    thinking_display = "\n".join(self.current_thinking[-20:])
                    yield history, thinking_display, f"Investigating... ({elapsed:.1f}s)"

        thread.join()

    def clear_chat(self) -> tuple[list, str, str]:
        """Clear the current chat session."""
        self.session = None
        self.current_thinking = []
        self.current_citations = []
        return [], "", "Chat cleared. Start a new conversation."

    def get_turn_details(self, turn_number: int) -> str:
        """Get detailed thinking logs for a specific turn."""
        if not self.session:
            return "No active session"

        turn_data = self.session.get_thinking_for_turn(turn_number)
        if not turn_data:
            return f"No data for turn {turn_number}"

        output = [
            f"## Turn {turn_number} Details",
            f"**Query:** {turn_data['query']}",
            f"**Time:** {turn_data['timestamp']}",
            "",
            "### Thinking Steps",
        ]
        output.extend(turn_data['thinking'])

        if turn_data['citations']:
            output.append("\n### Citations")
            output.extend(turn_data['citations'])

        return "\n".join(output)


def browse_folder() -> str:
    """Open native folder picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        folder_path = filedialog.askdirectory(
            title="Select Document Repository",
            initialdir=os.path.expanduser("~")
        )

        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"Folder picker error: {e}")
        return ""


def create_chat_app(api_key: Optional[str] = None) -> gr.Blocks:
    """Create the multi-turn chat Gradio application."""
    app = ChatApp(api_key=api_key)

    with gr.Blocks(
        title="Irys RLM - Legal Document Chat",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Irys RLM - Legal Document Chat")
        gr.Markdown(
            "Select a document repository and have a conversation about it. "
            "Ask follow-up questions - the system remembers your conversation."
        )

        with gr.Row():
            # Left column: Chat
            with gr.Column(scale=2):
                # Repository selector
                with gr.Row():
                    repo_path = gr.Textbox(
                        label="Repository Path",
                        placeholder="Select a folder containing legal documents...",
                        scale=4,
                    )
                    browse_btn = gr.Button("Browse", scale=1)

                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about the documents...",
                        scale=4,
                        lines=2,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")

            # Right column: Status and thinking
            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

                with gr.Accordion("Thinking Trace (Current Turn)", open=True):
                    thinking = gr.Textbox(
                        label="Investigation Steps",
                        interactive=False,
                        lines=20,
                        autoscroll=True,
                    )

        # Example queries
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["What are the main claims in this dispute?"],
                ["What is the timeline of key events?"],
                ["Who are the parties involved?"],
                ["What damages are being claimed?"],
                ["Can you explain more about the breach allegations?"],  # Follow-up style
            ],
            inputs=[msg],
        )

        # Event handlers
        browse_btn.click(
            fn=browse_folder,
            outputs=[repo_path],
        )

        # Submit on button click
        submit_btn.click(
            fn=app.chat,
            inputs=[msg, chatbot, repo_path],
            outputs=[chatbot, thinking, status],
        ).then(
            fn=lambda: "",  # Clear input after submit
            outputs=[msg],
        )

        # Submit on Enter
        msg.submit(
            fn=app.chat,
            inputs=[msg, chatbot, repo_path],
            outputs=[chatbot, thinking, status],
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        # Clear chat
        clear_btn.click(
            fn=app.clear_chat,
            outputs=[chatbot, thinking, status],
        )

    return demo


def main():
    """Run the chat application."""
    import argparse

    parser = argparse.ArgumentParser(description="Irys RLM Chat UI")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--port", type=int, default=7863, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No GEMINI_API_KEY provided")

    app = create_chat_app(api_key=api_key)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
