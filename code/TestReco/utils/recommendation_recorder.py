"""
Recommendation recorder for LLM agents.

This module provides tools to record recommendation information from LLM agents
using a decorator pattern for flexible addition and removal of recording functionality.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional

from generators.model import Message


class RecommendationRecorder:
    """
    Decorator class for recording recommendation information from LLM agents.

    This class can be used to wrap agent.select_action to record:
    - Recommendation prompt
    - LLM response
    - Recommended difficulty
    - Recommended skill
    - Reasoning
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the recommendation recorder.

        Args:
            enabled: Whether recording is enabled by default
        """
        self.enabled = enabled
        self.recordings = []

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation.

        Args:
            func: The function to wrap (typically agent.select_action)

        Returns:
            Wrapped function that records recommendation information
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the agent instance (first argument)
            agent = args[0]

            # Call the original function
            result = func(*args, **kwargs)

            if self.enabled:
                # Extract recommendation info from the result
                if isinstance(result, dict) and "recommendation_info" in result:
                    # Format from SimpleLLMAgent
                    recording = result["recommendation_info"]
                else:
                    # Fallback: direct extraction
                    recording = {
                        "recommendation_prompt": self._get_prompt(
                            agent.model.messages
                            if hasattr(agent.model, "messages")
                            else []
                        ),
                        "recommendation_response": self._get_response(agent.model),
                        "recommended_difficulty": self._get_difficulty(agent.model),
                        "recommended_skill": self._get_skill(agent.model),
                        "reasoning": self._get_reasoning(agent.model),
                    }

                # Add recording to the result if it's a dictionary
                if isinstance(result, dict):
                    result.update(recording)
                elif hasattr(result, "__dict__"):
                    for key, value in recording.items():
                        setattr(result, key, value)

                # Store the recording
                self.recordings.append(recording)

            return result

        return wrapper

    def _get_prompt(self, messages: list) -> Optional[str]:
        """Extract the prompt from messages."""
        if not messages:
            return None
        # Get the last user message
        for msg in reversed(messages):
            if isinstance(msg, Message) and msg.role == "user":
                return msg.content
        return None

    def _get_response(self, model) -> Optional[str]:
        """Extract the response from the model."""
        if hasattr(model, "last_response"):
            return model.last_response
        return None

    def _get_difficulty(self, model) -> Optional[int]:
        """Extract the recommended difficulty from the response."""
        response = self._get_response(model)
        if not response:
            return None
        try:
            # Look for patterns like "difficulty: 2" or "difficulty:2"
            import re

            match = re.search(r"difficulty:\s*(\d+)", response)
            if match:
                return int(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None

    def _get_skill(self, model) -> Optional[str]:
        """Extract the recommended skill from the response."""
        response = self._get_response(model)
        if not response:
            return None
        try:
            # Look for patterns like "skill: Math" or "skill:Math"
            import re

            match = re.search(r"skill:\s*(.+)", response)
            if match:
                return match.group(1).strip().strip("\"'")
        except (ValueError, AttributeError):
            pass
        return None

    def _get_reasoning(self, model) -> Optional[str]:
        """Extract the reasoning from the response."""
        response = self._get_response(model)
        if not response:
            return None
        try:
            # Look for patterns like "reasoning: ..." or "reasoning:..."
            import re

            match = re.search(r"reasoning:\s*(.+)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        except (ValueError, AttributeError):
            pass
        return None

    def get_recordings(self) -> list:
        """Get all recorded recommendations."""
        return self.recordings

    def clear_recordings(self):
        """Clear all recorded recommendations."""
        self.recordings = []

    def enable(self):
        """Enable recording."""
        self.enabled = True

    def disable(self):
        """Disable recording."""
        self.enabled = False
