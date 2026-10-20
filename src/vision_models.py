"""Vision models on OpenRouter for the agents that look at images (view analyses, path agent, corner check,
the validators' looking step). All take images and support tool calls and JSON replies. Prices are USD per
million tokens (input / output), from OpenRouter's model list on 2026-10-19.

Set one of them as VISION_MODEL in main.py.
"""

VISION_MODELS = {
    # vision-language models built for images
    "qwen/qwen3-vl-30b-a3b-instruct": "Qwen3-VL 30B, open weights; 0.15 / 0.60",
    "qwen/qwen3-vl-8b-instruct": "Qwen3-VL 8B, open weights, smaller; 0.25 / 0.75",
    "z-ai/glm-4.6v": "GLM-4.6V, open weights; 0.30 / 0.90",
    "nvidia/nemotron-3.5-super-vl": "Nemotron 3.5 Super VL, open weights; 0.10 / 0.40",
    "inclusionai/ling-3.0-flash-vl": "Ling 3.0 Flash VL, open weights, cheapest; 0.06 / 0.18",
    # multimodal general model, a step up in quality
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite; 0.25 / 1.50",
}
