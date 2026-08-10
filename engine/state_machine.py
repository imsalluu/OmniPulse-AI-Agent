class SalesStateMachine:
    def __init__(self):
        self.phases = ["Hook", "Discovery", "Objection", "Closing"]
        
    def determine_phase(self, history_text, current_state):
        """
        Simple logic to transition between sales phases based on keywords.
        In a production app, this would be handled by an LLM.
        """
        text = history_text.lower()
        if any(word in text for word in ["price", "cost", "expensive", "busy", "think about"]):
            return "Objection"
        if any(word in text for word in ["quote", "meeting", "sign up", "ready"]):
            return "Closing"
        if len(history_text.split()) > 50:
            return "Discovery"
        
        return current_state