class ForwardHook:
    """Hook for collecting the final output of TransformerDecoderLayer"""
    def __init__(self):
        self.collected_outputs = []
    
    def forward_hook_function(self, module, input, output):
        self.collected_outputs.append(output[0].detach())

    def clear(self):
        self.collected_outputs.clear()
