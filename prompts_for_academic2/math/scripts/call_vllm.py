
def call_vllm_api(
    system_instruction: str | None,
    contents: list,
    verbose: bool = True
) -> str | None:
    """
    使用 SDK 调用 Gemini API，启用流式传输和思考过程。
    
    Args:
        system_instruction: 系统提示词。
        contents: 用户/模型的对话回合列表。
        verbose: 是否打印思考过程和 token 使用情况。

    Returns:
        生成的文本内容，如果失败则返回 None。
    """
    if verbose:
        print("--- [Calling VLLM] ---")

    sampling_params = SamplingParams(
            temperature=0.1,              # randomness of the sampling
            min_p=0.05,
            top_p=0.9,
            skip_special_tokens=True,     # Whether to skip special tokens in the output
            max_tokens=32768 ,
            # stop=["</think>"]
        )
    GENERATION_CONFIG = types.GenerateContentConfig(
        temperature=0.1,
        top_p=1.0, 
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=32768
        ),
        system_instruction=system_instruction
    )

    try:
        if verbose:
            print("🧠 Thinking process:")
            print("💭 ", end='', flush=True)

        full_answer = ""
        thinking_shown = False
        final_usage_metadata = None

        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=GENERATION_CONFIG
        )

        for chunk in response_stream:
            if chunk.usage_metadata:
                final_usage_metadata = chunk.usage_metadata

            if not chunk.candidates:
                continue
            
            content_obj = chunk.candidates[0].content
            if not content_obj or not getattr(content_obj, "parts", None):
                continue

            for part in content_obj.parts:
                if part.thought:
                    if not thinking_shown:
                        print()
                        thinking_shown = True
                    print(f"🤔 {part.text}", end='', flush=True)
                elif part.text:
                    if thinking_shown:
                        print("\n\n📝 Answer:")
                        thinking_shown = False
                    print(part.text, end='', flush=True)
                    full_answer += part.text
        
        print() # 确保在流式输出后换行

        if verbose and final_usage_metadata:
            print(f"\nThinking Tokens: {final_usage_metadata.thoughts_token_count}")
            print(f"Answer Tokens: {final_usage_metadata.candidates_token_count}")
            print(f"Prompt Tokens: {final_usage_metadata.prompt_token_count}")
            print(f"Total Tokens: {final_usage_metadata.total_token_count}")
        
        if verbose:
            print("--- [API Call Completed] ---\n")
        
        return full_answer if full_answer else None

    except Exception as e:
        print(f"\nSDK API call failed: \n{e}")
        return None