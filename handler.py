import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams
import runpod

MODEL_PATH = "/models"

print("Initializing vLLM Engine...")
try:
    llm = LLM(model=MODEL_PATH)
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    raise

sampling_params = SamplingParams(max_tokens=512, temperature=0.5)
print("Engine Ready.")

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")

    try:
        outputs = llm.generate([prompt], sampling_params)[0]
        response = outputs.outputs[0].text

        # Clean up the output slightly if it repeats the prompt
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return {"output": {"response": response}}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "max_concurrency": 3,
        "timeout": 90,
        "retry_count": 1
    })
