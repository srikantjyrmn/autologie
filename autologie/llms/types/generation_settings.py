"""Generation Settings"""
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field
class LlamacppGenerationSettings(BaseModel):
    """
    Settings for generating completions using the Llama.cpp server.
    Args:
        temperature (float): Controls the randomness of the generated completions. Higher values make the output more random.
        top_k (int): Controls the diversity of the top-k sampling. Higher values result in more diverse completions.
        top_p (float): Controls the diversity of the nucleus sampling. Higher values result in more diverse completions.
        min_p (float): Minimum probability for nucleus sampling. Lower values result in more focused completions.
        n_predict (int): Number of completions to predict. Set to -1 to use the default value.
        n_keep (int): Number of completions to keep. Set to 0 for all predictions.
        stream (bool): Enable streaming for long completions.
        additional_stop_sequences (List[str]): List of stop sequences to finish completion generation. The official stop sequences of the model get added automatically.
        tfs_z (float): Controls the temperature for top frequent sampling.
        typical_p (float): Typical probability for top frequent sampling.
        repeat_penalty (float): Penalty for repeating tokens in completions.
        repeat_last_n (int): Number of tokens to consider for repeat penalty.
        penalize_nl (bool): Enable penalizing newlines in completions.
        presence_penalty (float): Penalty for presence of certain tokens.
        frequency_penalty (float): Penalty based on token frequency.
        penalty_prompt (Union[None, str, List[int]]): Prompts to apply penalty for certain tokens.
        mirostat_mode (int): Mirostat level.
        mirostat_tau (float): Mirostat temperature.
        mirostat_eta (float): Mirostat eta parameter.
        seed (int): Seed for randomness. Set to -1 for no seed.
        ignore_eos (bool): Ignore end-of-sequence token.
    """
    max_tokens: int = 4096
    temperature: float = 0.1
    logprobs: int = None
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    n_keep: int = 0
    stream: bool = False
    additional_stop_sequences: List[str] = ['<|im_end|>']
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = -1
    penalize_nl: bool = False
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.1
    penalty_prompt: Union[None, str, List[int]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    cache_prompt: bool = True
    seed: int = -1
    ignore_eos: bool = False
    samplers: List[str] = None
    grammar: str | None = None

class OpenaiGenerationSettings(BaseModel):
    """Generation Settings for OpenAI"""
    max_tokens: int = None
    logprobs: bool = True
    top_logprobs: int = None
    temperature: float = 0.1
    top_p: float = Field(default = 0.95, ge=0, le=1)
    min_p: float = Field(default = 0.95, ge=0, le=1)
    stop: List[str] = None
    stream: bool = False
    presence_penalty: float = Field(
        default = 0, ge=-2, le=2,
        description = """Positive values penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.""")

    frequency_penalty: float = Field(
        default = 0, ge=-2, le=2,
        description = """Positive values penalize new tokens based on their existing frequency in the text so far,
        decreasing the model's likelihood to repeat the same line verbatim.
        """
    )
    logit_bias: Optional[str] = None #TODO
    seed: Optional[int] = None
    n: int = 1
    user: Optional[str] = None
    top_k: int = Field(
        description = """Limit the next token selection to the K most probable tokens.
        Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.
        Default40""", 
        default = 40
    )
    repeat_penalty: int = Field(
        description = """A penalty applied to each token that is already generated. This helps prevent the model from repeating itself. Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    Default1.1""",
        default = 1.1, ge = 0)
    logit_bias_type : Optional[Literal["input_ids", "tokens"]] = None
    mirostat_mode:int = Field(
        description = """Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)
        Default0""",
        default = 0, le = 2, ge = 0)
    mirostat_tau:float = Field(
        description = """Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text""",
        default = 5, le = 10, ge = 0)
    mirostat_eta:float =  Field(
        description = """Mirostat learning rate""",
        default = 0.1, le = 1, ge = 0.001)
    grammar: Optional[str] = None


class OllamaGenerationSettings(BaseModel):
    """
    Ref: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    """
    mirostat: Optional[int] = Field(
        description = """Enable Mirostat sampling for controlling perplexity.
        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)	
        Example: int	mirostat 0""",
        default = 0)
    mirostat_eta: Optional[float] = Field(
        description = """Influences how quickly the algorithm
        responds to feedback from the generated text.
        A lower learning rate will result in slower adjustments, 
        while a higher learning rate will make the algorithm more responsive. 
        (Default: 0.1)	float	
        mirostat_eta 0.1""",
        default = 0.1)
    mirostat_tau: float = Field(
        description = """Controls the balance between coherence and diversity of the output.
        A lower value will result in more focused and coherent text. (Default: 5.0)	float	
        mirostat_tau 5.0""",
        default = 5)
    num_ctx : Optional[int] = Field(description = """Sets the size of the context window
        used to generate the next token. (Default: 2048)	int	
        num_ctx 4096""",
        default = 4096)
    repeat_last_n :int = Field(
        description = """Sets how far back for the model to look back to prevent repetition
        (Default: 64, 0 = disabled, -1 = num_ctx)	int	repeat_last_n 64""",
        default = 64)
    repeat_penalty: float = Field(description = """Sets how strongly to penalize repetitions.
        A higher value (e.g., 1.5) will penalize repetitions more strongly, 
        while a lower value (e.g., 0.9) will be more lenient. 
        (Default: 1.1)	
        float	repeat_penalty 1.1""",
        default = 1.1)
    temperature: float	 = Field(
        description = """The temperature of the model.
        Increasing the temperature will make the model answer more creatively. 
        (Default: 0.8)	float	temperature 0.7""",
        default = 0.1)
    seed: int = Field(
        description = """Sets the random number seed to use for generation.
        Setting this to a specific number will make the model 
        generate the same text for the same prompt. 
        (Default: 0)	int	seed 42""",
        default = 0)
    stop: Optional[List[str]] = Field(
        description = """Sets the stop sequences to use.
        When this pattern is encountered the LLM will stop generating text and return. 
        Multiple stop patterns may be set by specifying multiple separate stop parameters in a model  file.	
        string	stop 'AI assistant:'""",
        default = ['<|im_end|>', '<|im_start|>'])
    tfs_z: float = Field(
        description = """Tail free sampling is used to reduce t
        he impact of less probable tokens from the output. 
        A higher value (e.g., 2.0) will reduce the impact more, 
        while a value of 1.0 disables this setting. 
        (default: 1)	float	tfs_z 1""",
        default = 1)
    num_predict: int = Field(description = """Maximum number of tokens to predict
                        when generating text. 
                        (Default: 128, 
                        -1 = infinite generation, 
                        -2 = fill context)	
                        int	num_predict 42""",
                        default = 1028)
    top_k: Optional[int] = Field(
        description = """	Reduces the probability of generating nonsense.
        A higher value (e.g. 100) will give more diverse answers, 
        while a lower value (e.g. 10) will be more conservative. 
        (Default: 40)	int	top_k 40""",
        default = 40)
    top_p: Optional[float] = Field(
        description = """	Works together with top-k.
        A higher value (e.g., 0.95) will lead to more diverse text, 
        while a lower value (e.g., 0.5) will generate more focused and conservative text. 
        (Default: 0.9)""",
        default=0.9 )

GenerationSettings = Union[
    LlamacppGenerationSettings,
    OllamaGenerationSettings,
    OpenaiGenerationSettings
]
