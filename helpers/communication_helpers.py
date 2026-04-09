from typing import Optional


VALID_MESSAGE_TYPES = frozenset({'ask_help', 'help', 'message'})


def _extract_message(msg, agent_id) -> Optional[dict]:
        """Extract structured data from a MATRX Message object."""
        from_id = getattr(msg, 'from_id', None)
        to_id = getattr(msg, 'to_id', None)
        content = getattr(msg, 'content', None)

        # Skip messages from self
        if from_id == agent_id:
            return None

        # Handle structured content (dict with message_type + text)
        if isinstance(content, dict):
            text = content.get('text', '')
            msg_type = content.get('message_type', 'message')
            if msg_type not in VALID_MESSAGE_TYPES:
                msg_type = 'message'
        elif isinstance(content, str):
            text = content
            msg_type = 'message'
        else:
            return None

        if not text:
            return None

        return {
            'from': from_id or 'unknown',
            'to': 'all' if to_id is None else (to_id if isinstance(to_id, str) else str(to_id)),
            'message_type': msg_type,
            'text': text,
        }