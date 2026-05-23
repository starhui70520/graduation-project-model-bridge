import onnxruntime_genai as og
import chatglm_cpp

def Phi3_Call(text, model):
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    prompt = f'<|user|>\n{text} <|end|>\n<|assistant|>'

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.try_use_cuda_graph_with_max_batch_size(1)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    output_text = ""  # 用于存储生成的文本

    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(new_token)  # 将生成的文本添加到字符串中
    except KeyboardInterrupt:
        output_text += "  --control+c pressed, aborting generation--"

    return output_text  # 返回完整的生成文本

def ChatGLM_Call(text, pipeline, flag, input_cache=""):
    if flag == "input":
        input_prompt = f"""请将目标输入的内容翻译成英文
        你的工作只是翻译，请确保翻译出来的内容有且只有英文，不能修改原有意思，输出格式如下:
        Question: {text}
        
        以下是具体输入:
        {text}
        """

        result = pipeline.chat([chatglm_cpp.ChatMessage(role="user", content=input_prompt)])
        output_content = result.content
        
        # 检查输出是否以"Question:"开头，如预期那样
        if output_content.startswith("Question:"):
            # 提取"Question:"之后的文本
            return output_content.split("Question:")[1].strip()
        else:
            # 递归调用函数，直到输出符合期望的格式
            return ChatGLM_Call(text, pipeline, flag)
        
    else:
        output_prompt = f"""请将用户回复的内容翻译成中文
        你的主要工作是翻译，将用户回复的内容除代码外完全转换成中文，假如输入的内容所表达的意思不清楚或表达不知道的
        假如你正巧知道，那么你有权用自己的知识来重述这段话，我会给到你提问时的内容，假如你也不知道，请做好翻译工作即可
        输出格式如下:
        Answer: {text}
        
        这是提问的提示内容：
        {input_cache}
        以下是具体输入:
        {text}
        """

        result = pipeline.chat([chatglm_cpp.ChatMessage(role="user", content=output_prompt)])
        output_content = result.content
        
        # 检查输出是否以"Answer:"开头，如预期那样
        if output_content.startswith("Answer:"):
            # 提取"Answer:"之后的文本
            return output_content.split("Answer:")[1].strip()
        else:
            # 递归调用函数，直到输出符合期望的格式
            return ChatGLM_Call(text, pipeline, flag)


def main():
    model = og.Model(f'directml\directml-int4-awq-block-128')
    pipeline = chatglm_cpp.Pipeline("chatglm-ggml.bin")

    Text = "你是谁"

    Chinese_Question = ChatGLM_Call(Text, pipeline, "input")
    print(Chinese_Question)
    English_Answer = Phi3_Call(Chinese_Question, model)
    print(English_Answer)
    Chinese_Answer = ChatGLM_Call(English_Answer, pipeline, "output", Text)
    print(Chinese_Answer)

if __name__ == "__main__":
    main()