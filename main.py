import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QLabel, QTextBrowser, QLineEdit, QPushButton, QWidget
from PySide6.QtCore import QThread, Signal
import onnxruntime_genai as og
import chatglm_cpp

class Phi3Worker(QThread):
    finished = Signal(str)
    def __init__(self):
        super().__init__()
        self.model = og.Model(f'directml\directml-int4-awq-block-128')

    def run(self):
        tokenizer = og.Tokenizer(self.model)
        tokenizer_stream = tokenizer.create_stream()

        prompt = f'<|user|>\n{self.text} <|end|>\n<|assistant|>'
        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.try_use_cuda_graph_with_max_batch_size(1)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)

        output_text = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output_text += tokenizer_stream.decode(new_token)

        self.finished.emit(output_text)

class ChatGLMWorker(QThread):
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self.pipeline = chatglm_cpp.Pipeline("chatglm-ggml.bin")

    def run(self):
        if self.flag == "input":
            input_prompt = f"""请将目标输入的内容翻译成英文
            你的工作只是翻译，请确保翻译出来的内容有且只有英文，不能修改原有意思，输出格式如下:
            Question: 你翻译的内容
            
            以下是具体输入:
            {self.text}
            """

            result = self.pipeline.chat([chatglm_cpp.ChatMessage(role="user", content=input_prompt)])
            output_content = result.content
            
            if output_content.startswith("Question:"):
                self.finished.emit(output_content.split("Question:")[1].strip())
            else:
                self.finished.emit(self.text)       
        else:
            output_prompt = f"""
            
            这是提问的提示内容：
            {self.input_cache}
            以下是具体输入,结合上面提问时的内容对将以下的英文内容翻译成中文，输出任何内容开头必须带上Answer: 
            {self.text}
            """

            result = self.pipeline.chat([chatglm_cpp.ChatMessage(role="user", content=output_prompt)])
            output_content = result.content
            
            if output_content.startswith("Answer:"):
                self.finished.emit(output_content.split("Answer:")[1].strip())
            else:
                self.finished.emit(self.text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("日志记录器")

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        left_widget = QLabel("保存日志：")
        self.left_text_browser = QTextBrowser()

        left_layout.addWidget(left_widget)
        left_layout.addWidget(self.left_text_browser)

        right_widget = QLabel("模型问答：")
        self.right_text_browser = QTextBrowser()
        self.line_edit = QLineEdit()
        send_button = QPushButton("发送")

        right_layout.addWidget(right_widget)
        right_layout.addWidget(self.right_text_browser)
        right_layout.addWidget(self.line_edit)
        right_layout.addWidget(send_button)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 7)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)

        self.chatglm_worker = ChatGLMWorker()
        self.phi3_worker = Phi3Worker()
        self.chatglm_worker.finished.connect(self.on_phi3_finished)
        send_button.clicked.connect(self.on_send_clicked)
        

    def on_send_clicked(self):
        self.input_text = self.line_edit.text()
        self.right_text_browser.clear()
        self.left_text_browser.append(f"用户输入: \n{self.input_text}")
        self.right_text_browser.append(f"用户输入: \n{self.input_text}")
        # 传入ChatGLM做中译英
        self.chatglm_worker.text = self.input_text
        self.chatglm_worker.flag = "input"
        self.chatglm_worker.input_cache = ''
        self.chatglm_worker.start()

    def on_phi3_finished(self, input_translated_text):
        self.input_translated_text = input_translated_text
        self.left_text_browser.append(f"用户输入转英文: \n{input_translated_text}")
        # 传入Phi3模型获取回复
        self.phi3_worker.text = input_translated_text
        self.phi3_worker.finished.disconnect()
        self.phi3_worker.finished.connect(self.on_chatglm_finished)
        self.phi3_worker.start()

    def on_chatglm_finished(self, phi3_output_text):
        self.left_text_browser.append(f"Phi回复: \n{phi3_output_text}")
        # 传入ChatGLM做英译中
        self.phi3_output_text = phi3_output_text
        self.chatglm_worker.text = phi3_output_text
        self.chatglm_worker.flag = "output"
        self.chatglm_worker.input_cache = self.input_text
        self.chatglm_worker.finished.disconnect()
        self.chatglm_worker.finished.connect(self.on_output)
        self.chatglm_worker.start()


    def on_output(self, output_translated_text):
        self.left_text_browser.append(f"ChatGLM翻译中文: \n{output_translated_text}")
        self.right_text_browser.append(f"回答: \n{output_translated_text}")
        self.chatglm_worker.finished.disconnect()
        self.chatglm_worker.finished.connect(self.on_phi3_finished)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
