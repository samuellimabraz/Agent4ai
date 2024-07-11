import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import requests
import gradio as gr


class ChatInterface:
    """
    Classe para gerenciar a interface do agente.

    Métodos:
    - response: Envia uma mensagem para o servidor e retorna a resposta do Agent.
    - clean_memory: Limpa a memória do agente.
    - launch_interface: Lança a interface de chat.
    """

    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"
        self.logs = []

    def response(self, message, history):
        """
        Envia uma mensagem para o servidor e retorna a resposta.

        Args:
            message (str): Mensagem a ser enviada.
            history (list): Histórico de mensagens.

        Returns:
            str: Resposta do servidor.
        """
        response = requests.post(
            f"{self.base_url}/query", json={"request": message}
        ).json()
        generation = response["generation"]
        history.append((message, generation))
        self.logs.append(response)
        return "", history

    def clean_memory(self):
        """
        Limpa a memória do servidor.

        Returns:
            str: Resposta do servidor após limpar a memória.
        """
        response = requests.post(f"{self.base_url}/clear", json={}).json()
        print(response)
        return response

    def launch_interface(self):
        """
        Lança a interface de chat usando Gradio.
        """
        with gr.Blocks(title="Agent") as demo:
            self.chatbot = gr.Chatbot(
                label="Agent",
                height=600,
                show_copy_button=True,
                show_share_button=True,
                layout="bubble",
            )

            self.msg = gr.Textbox(
                placeholder="Me pergunte algo!",
                container=False,
                scale=7,
            )
            self.msg.submit(
                self.response, [self.msg, self.chatbot], [self.msg, self.chatbot]
            )

            clear = gr.ClearButton([self.msg, self.chatbot], value="Limpar memória")
            clear.click(self.clean_memory, inputs=[], outputs=[])

        demo.launch()


if __name__ == "__main__":
    chat = ChatInterface()
    chat.launch_interface()
