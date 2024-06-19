from autologie.llms import LlmClient, FunctionCallingLlm, UnstructuredLlm, StructuredLlm

class FunctionCallingAgent:
    llm: FunctionCallingLlm = FunctionCallingLlm()

    def __init__(self, llm: FunctionCallingLlm, **kwargs):
        pass

class StructuredResponseAgent:
    llm: FunctionCallingLlm = StructuredLlm()

    def __init__(self, llm: FunctionCallingLlm, **kwargs):
        pass

class RolePlayingAgent:
    llm: UnstructuredLlm = UnstructuredLlm()